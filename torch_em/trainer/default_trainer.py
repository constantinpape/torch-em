from __future__ import annotations

import contextlib
import inspect
import os
import time
import warnings
from collections import OrderedDict
from importlib import import_module
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.cuda.amp as amp
from tqdm import tqdm

from .tensorboard_logger import TensorboardLogger
from .wandb_logger import WandbLogger
from ..util import auto_compile, get_constructor_arguments, is_compiled


class DefaultTrainer:
    """Trainer class for 2d/3d training on a single GPU."""

    def __init__(
        self,
        name: Optional[str],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss,
        optimizer,
        metric,
        device: Union[str, torch.device],
        lr_scheduler=None,
        log_image_interval=100,
        mixed_precision=True,
        early_stopping=None,
        logger=TensorboardLogger,
        logger_kwargs: Optional[Dict[str, Any]] = None,
        id_: Optional[str] = None,
        save_root: Optional[str] = None,
        compile_model: Optional[Union[bool, str]] = None,
    ):
        if name is None and not issubclass(logger, WandbLogger):
            raise TypeError("Name cannot be None if not using the WandbLogger")

        if not all(hasattr(loader, "shuffle") for loader in [train_loader, val_loader]):
            raise ValueError(f"{self.__class__} requires each dataloader to have 'shuffle' attribute.")

        self._generate_name = name is None
        self.name = name
        self.id_ = id_ or name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.log_image_interval = log_image_interval
        self.save_root = save_root
        self.compile_model = compile_model

        self._iteration = 0
        self._epoch = 0
        self._best_epoch = 0

        self.mixed_precision = mixed_precision
        self.early_stopping = early_stopping

        self.scaler = amp.GradScaler() if mixed_precision else None

        self.logger_class = logger
        self.logger_kwargs = logger_kwargs
        self.log_image_interval = log_image_interval

    @property  # because the logger may generate and set trainer.id on logger.__init__
    def checkpoint_folder(self):
        assert self.id_ is not None
        # save_root enables saving the checkpoints somewhere else than in the local
        # folder. This is handy for filesystems with limited space, where saving the checkpoints
        # and log files can easily lead to running out of space.
        save_root = getattr(self, "save_root", None)
        return os.path.join("./checkpoints", self.id_) if save_root is None else\
            os.path.join(save_root, "./checkpoints", self.id_)

    @property
    def iteration(self):
        return self._iteration

    @property
    def epoch(self):
        return self._epoch

    class Deserializer:
        """Determines how to deserialize the trainer kwargs from serialized 'init_data'

        Examples:
            To extend the initialization process you can inherite from this Deserializer in an inherited Trainer class.
            Note that `DefaultTrainer.Deserializer.load_generic()` covers most cases already.

            This example adds `the_answer` kwarg, which requires 'calculations' upon initialization:
            >>> class MyTrainer(DefaultTrainer):
            >>>     def __init__(self, *args, the_answer: int, **kwargs):
            >>>         super().__init__(*args, **kwargs)
            >>>         self.the_answer = the_answer  # this allows the default Serializer to save the new kwarg,
            >>>                                       # see DefaultTrainer.Serializer
            >>>
            >>>     class Deserializer(DefaultTrainer.Deserializer):
            >>>         def load_the_answer(self):
            >>>             generic_answer = self.init_data["the_answer"]
            >>>             # (device dependent) special deserialization
            >>>             if self.trainer_kwargs["device"].type == "cpu":  # accessing previously deserialized kwarg
            >>>                 self.trainer_kwargs["the_answer"] = generic_answer + 1
            >>>             else:
            >>>                 self.trainer_kwargs["the_answer"] = generic_answer * 2
        """

        def __init__(self, init_data: dict, save_path: str, device: Union[str, torch.device]):
            self.init_data = init_data
            self.save_path = save_path
            # populate with deserialized trainer kwargs during deserialization; possibly overwrite 'device'
            self.trainer_kwargs: Dict[str, Any] = dict(
                device=torch.device(self.init_data["device"]) if device is None else torch.device(device)
            )

        def load(self, kwarg_name: str, optional):
            """`optional` is True if self.trainer.__class__.__init__ specifies a default value for 'kwarg_name'"""

            if kwarg_name == "device":
                pass  # deserialized in __init__
            elif kwarg_name.endswith("_loader"):
                self.load_data_loader(kwarg_name, optional)
            else:
                load = getattr(self, f"load_{kwarg_name}", self.load_generic)
                load(kwarg_name, optional=optional)

        def load_data_loader(self, loader_name, optional) -> None:
            ds = self.init_data.get(loader_name.replace("_loader", "_dataset"))
            if ds is None and optional:
                return

            loader_kwargs = self.init_data[f"{loader_name}_kwargs"]
            loader = torch.utils.data.DataLoader(ds, **loader_kwargs)
            # monkey patch shuffle loader_name to the loader
            loader.shuffle = loader_kwargs.get("shuffle", False)
            self.trainer_kwargs[loader_name] = loader

        def load_generic(
            self,
            kwarg_name: str,
            *dynamic_args,
            optional: bool,
            only_class: bool = False,
            dynamic_kwargs: Optional[Dict[str, Any]] = None,
        ) -> None:
            if kwarg_name in self.init_data:
                self.trainer_kwargs[kwarg_name] = self.init_data[kwarg_name]
                return

            this_cls = self.init_data.get(f"{kwarg_name}_class", None)
            if this_cls is None:
                if optional:
                    return
                else:
                    raise RuntimeError(f"Could not find init data for {kwarg_name} in {self.save_path}")

            assert isinstance(this_cls, str), this_cls
            assert "." in this_cls, this_cls
            cls_p, cls_m = this_cls.rsplit(".", 1)
            this_cls = getattr(import_module(cls_p), cls_m)
            if only_class:
                self.trainer_kwargs[kwarg_name] = this_cls
            else:
                self.trainer_kwargs[kwarg_name] = this_cls(
                    *dynamic_args, **self.init_data.get(f"{kwarg_name}_kwargs", {}), **(dynamic_kwargs or {})
                )

        def load_name(self, kwarg_name: str, optional: bool):
            self.trainer_kwargs[kwarg_name] = os.path.split(os.path.dirname(self.save_path))[1]

        def load_optimizer(self, kwarg_name: str, optional: bool):
            self.load_generic(kwarg_name, self.trainer_kwargs["model"].parameters(), optional=optional)

        def load_lr_scheduler(self, kwarg_name: str, optional: bool):
            self.load_generic(kwarg_name, self.trainer_kwargs["optimizer"], optional=optional)

        # todo: remove and rename kwarg 'logger' to 'logger_class'
        def load_logger(self, kwarg_name: str, optional: bool):
            assert kwarg_name == "logger"
            self.load_generic("logger", optional=optional, only_class=True)

    @staticmethod
    def _get_save_dict(save_path, device):
        if not os.path.exists(save_path):
            raise ValueError(f"Cannot find checkpoint {save_path}")
        return torch.load(save_path, map_location=device)

    @classmethod
    def from_checkpoint(cls, checkpoint_folder, name="best", device=None):
        save_path = os.path.join(checkpoint_folder, f"{name}.pt")
        # make sure the correct device is set if we don't have access to CUDA
        if not torch.cuda.is_available():
            device = "cpu"
        save_dict = cls._get_save_dict(save_path, device)
        deserializer = cls.Deserializer(save_dict["init"], save_path, device)

        has_kwargs = False
        deserialized = []
        for name, parameter in inspect.signature(cls).parameters.items():
            if name == "kwargs":
                has_kwargs = True
                continue
            deserializer.load(name, optional=parameter.default is not inspect.Parameter.empty)
            deserialized.append(name)

        # to deserialze kwargs we can't rely on inspecting the signature, so we
        # go through the remaning kwarg names in init data instead
        if has_kwargs:
            kwarg_names = list(set(deserializer.init_data.keys()) - set(deserialized))
            for name in kwarg_names:
                if name.endswith("_kwargs"):
                    continue
                elif name.endswith("_dataset"):
                    deserializer.load(name.replace("dataset", "loader"), optional=False)
                elif name.endswith("_class"):
                    deserializer.load(name.replace("_class", ""), optional=False)
                else:
                    deserializer.load(name, optional=False)

        trainer = cls(**deserializer.trainer_kwargs)
        trainer._initialize(0, save_dict)
        trainer._is_initialized = True
        return trainer

    class Serializer:
        """Implements how to serialize trainer kwargs from a trainer instance

        Examples:
            To extend the serialization process you can inherite from this Serializer in a derived Trainer class.
            Note that the methods `dump_generic_builtin()`, `dump_generic_class()` and `dump_generic_instance()`
            called by the `dump()` method when appropriate cover most cases already.

            This example adds `the_answer` kwarg, which requires extra steps on dumping only because we don't keep a
            'the_answer' attribute:
            >>> class MyTrainer(DefaultTrainer):
            >>>     def __init__(self, *args, the_answer: int, **kwargs):
            >>>         super().__init__(*args, **kwargs)
            >>>         # self.the_answer = the_answer  # this would allow the default Serializer to save the new kwarg,
            >>>         # but let's make things more interesting...
            >>>         self.the = the_answer // 10
            >>>         self.answer = the_answer % 10
            >>>
            >>>     class Serializer(DefaultTrainer.Serializer):
            >>>         trainer: MyTrainer
            >>>         def dump_the_answer(self, kwarg_name: str) -> None:  # custom dump method for 'the_answer' kwarg
            >>>             assert kwarg_name == "the_answer"
            >>>             # populate self.init_data with the serialized data required by Deserializer
            >>>             # to restore the trainer kwargs
            >>>             self.init_data["the_answer"] = self.trainer.the * 10 + self.trainer.answer

            This example with both Serializer and Deserializer adds `the_answer` kwarg,
            while saving it in two separate entries 'the' and 'answer'
            >>> class MyTrainer(DefaultTrainer):
            >>>     def __init__(self, *args, the_answer: int, **kwargs):
            >>>         super().__init__(*args, **kwargs)
            >>>         self.the_answer = the_answer
            >>>
            >>>     class Serializer(DefaultTrainer.Serializer):
            >>>         trainer: MyTrainer
            >>>         def dump_the_answer(self, kwarg_name: str):
            >>>             assert kwarg_name == "the_answer"
            >>>             self.init_data.update({
            >>>                 "the": self.trainer.the_answer // 10,
            >>>                 "answer": self.trainer.the_answer % 10
            >>>             })
            >>>
            >>>     class Deserializer(DefaultTrainer.Deserializer):
            >>>         def load_the_answer(self, kwarg_name: str, optional: bool):
            >>>             assert kwarg_name == "the_answer"
            >>>             # 'optional' is True if MyTrainer.__init__ specifies a default value for 'kwarg_name'
            >>>             self.trainer_kwargs[kwarg_name] = self.init_data["the"] * 10 + self.init_data["answer"]
        """

        def __init__(self, trainer: DefaultTrainer):
            self.trainer = trainer
            self.init_data = {}  # to be populated during serialization process

        def dump(self, kwarg_name: str) -> None:
            dumper = getattr(self, f"dump_{kwarg_name}", None)
            if dumper is not None:
                dumper(kwarg_name)
            elif kwarg_name.endswith("_loader"):
                self.dump_data_loader(kwarg_name)
            elif kwarg_name.endswith("_class"):
                self.dump_generic_class(kwarg_name)
            elif not hasattr(self.trainer, kwarg_name):
                raise AttributeError(
                    f"{self.trainer.__class__} missing attribute '{kwarg_name}' "
                    f"or special dump method {self.trainer.__class__}.Serializer.dump_{kwarg_name}()"
                )
            else:
                assert hasattr(self.trainer, kwarg_name)
                obj = getattr(self.trainer, kwarg_name)
                if obj is None or type(obj) in (
                    bool,
                    bytearray,
                    bytes,
                    dict,
                    float,
                    frozenset,
                    int,
                    list,
                    set,
                    str,
                    tuple,
                ):
                    self.dump_generic_builtin(kwarg_name)
                else:
                    self.dump_generic_instance(kwarg_name)

        def dump_generic_builtin(self, kwarg_name: str) -> None:
            assert hasattr(self.trainer, kwarg_name)
            self.init_data[kwarg_name] = getattr(self.trainer, kwarg_name)

        def dump_generic_class(self, kwarg_name: str) -> None:
            assert hasattr(self.trainer, kwarg_name)
            assert kwarg_name.endswith("_class")
            obj = getattr(self.trainer, kwarg_name)
            self.init_data[kwarg_name] = None if obj is None else f"{obj.__module__}.{obj.__name__}"

        def dump_generic_instance(self, kwarg_name: str) -> None:
            assert hasattr(self.trainer, kwarg_name)
            instance = getattr(self.trainer, kwarg_name)
            self.init_data.update(
                {
                    f"{kwarg_name}_class": f"{instance.__class__.__module__}.{instance.__class__.__name__}",
                    f"{kwarg_name}_kwargs": get_constructor_arguments(instance),
                }
            )

        def dump_device(self, kwarg_name: str):
            assert hasattr(self.trainer, kwarg_name)
            self.init_data[kwarg_name] = str(getattr(self.trainer, kwarg_name))

        def dump_data_loader(self, kwarg_name: str) -> None:
            assert hasattr(self.trainer, kwarg_name)
            loader = getattr(self.trainer, kwarg_name)
            if loader is None:
                return
            self.init_data.update(
                {
                    f"{kwarg_name.replace('_loader', '_dataset')}": loader.dataset,
                    f"{kwarg_name}_kwargs": get_constructor_arguments(loader),
                }
            )

        def dump_logger(self, kwarg_name: str):  # todo: remove and rename kwarg 'logger' to 'logger_class'
            self.dump_generic_class(f"{kwarg_name}_class")

        def dump_model(self, kwarg_name: str):
            if is_compiled(self.trainer.model):
                self.init_data.update(
                    {
                        "model_class": self.trainer._model_class,
                        "model_kwargs": self.trainer._model_kwargs,
                    }
                )
            else:
                self.dump_generic_instance("model")

    def _build_init(self) -> Dict[str, Any]:
        serializer = self.Serializer(self)
        for name in inspect.signature(self.__class__).parameters:
            # special rules to serialize kwargs
            # if a trainer class inherits from DefaultTrainer and has **kwargs
            # they need to be saved in self._kwargs
            if name == "kwargs":
                if not hasattr(self, "_kwargs"):
                    msg = "The trainer class has **kwargs in its signature, but is missing the _kwargs attribute. " +\
                          "Please add self._kwargs to its __init__ function"
                    raise RuntimeError(msg)
                kwargs = getattr(self, "_kwargs")
                for kwarg_name in kwargs:
                    serializer.dump(kwarg_name)
                continue
            serializer.dump(name)

        return serializer.init_data

    def _initialize(self, iterations, load_from_checkpoint):
        assert self.train_loader is not None
        assert self.val_loader is not None
        assert self.model is not None
        assert self.loss is not None
        assert self.optimizer is not None
        assert self.metric is not None
        assert self.device is not None

        if load_from_checkpoint is not None:
            self.load_checkpoint(load_from_checkpoint)

        self.max_iteration = self._iteration + iterations
        epochs = int(np.ceil(float(iterations) / len(self.train_loader)))
        self.max_epoch = self._epoch + epochs

        if not getattr(self, "_is_initialized", False):
            # check if we compile the model (only supported by pytorch 2)
            # to enable (de)serialization of compiled models, we keep track of the model class and kwargs
            if is_compiled(self.model):
                warnings.warn(
                    "You have passed a compiled model to the trainer."
                    "It will not be possible to (de)serialize the trainer with it."
                    "If you want to be able to do this please pass the normal model."
                    "It can be automatically compiled by setting 'compile_model' to True"
                )
            self._model_class = f"{self.model.__class__.__module__}.{self.model.__class__.__name__}"
            self._model_kwargs = get_constructor_arguments(self.model)
            self.model = auto_compile(self.model, self.compile_model)

            self.model.to(self.device)
            self.loss.to(self.device)

            # this saves all the information that is necessary
            # to fully load the trainer from the checkpoint
            self.init_data = self._build_init()

            if self.logger_class is None:
                self.logger = None
            else:
                # may set self.name if self.name is None
                save_root = getattr(self, "save_root", None)
                self.logger = self.logger_class(self, save_root, **(self.logger_kwargs or {}))

            os.makedirs(self.checkpoint_folder, exist_ok=True)

        best_metric = np.inf
        return best_metric

    def save_checkpoint(self, name, best_metric, **extra_save_dict):
        save_path = os.path.join(self.checkpoint_folder, f"{name}.pt")
        extra_init_dict = extra_save_dict.pop("init", {})
        save_dict = {
            "iteration": self._iteration,
            "epoch": self._epoch,
            "best_epoch": self._best_epoch,
            "best_metric": best_metric,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "init": self.init_data | extra_init_dict,
        }
        save_dict.update(**extra_save_dict)
        if self.scaler is not None:
            save_dict.update({"scaler_state": self.scaler.state_dict()})
        if self.lr_scheduler is not None:
            save_dict.update({"scheduler_state": self.lr_scheduler.state_dict()})
        torch.save(save_dict, save_path)

    def load_checkpoint(self, checkpoint="best"):
        if isinstance(checkpoint, str):
            save_path = os.path.join(self.checkpoint_folder, f"{checkpoint}.pt")
            if not os.path.exists(save_path):
                warnings.warn(f"Cannot load checkpoint. {save_path} does not exist.")
                return
            save_dict = torch.load(save_path)
        elif isinstance(checkpoint, dict):
            save_dict = checkpoint
        else:
            raise RuntimeError

        self._iteration = save_dict["iteration"]
        self._epoch = save_dict["epoch"]
        self._best_epoch = save_dict["best_epoch"]
        self.best_metric = save_dict["best_metric"]

        model_state = save_dict["model_state"]
        # to enable loading compiled models
        compiled_prefix = "_orig_mod."
        model_state = OrderedDict(
            [(k[len(compiled_prefix):] if k.startswith(compiled_prefix) else k, v) for k, v in model_state.items()]
        )
        self.model.load_state_dict(model_state)
        # we need to send the network to the device before loading the optimizer state!
        self.model.to(self.device)

        self.optimizer.load_state_dict(save_dict["optimizer_state"])
        if self.scaler is not None:
            self.scaler.load_state_dict(save_dict["scaler_state"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(save_dict["scheduler_state"])

        return save_dict

    def fit(self, iterations, load_from_checkpoint=None):
        best_metric = self._initialize(iterations, load_from_checkpoint)
        print(
            "Start fitting for",
            self.max_iteration - self._iteration,
            "iterations / ",
            self.max_epoch - self._epoch,
            "epochs",
        )
        print("with", len(self.train_loader), "iterations per epoch")

        if self.mixed_precision:
            train_epoch = self._train_epoch_mixed
            validate = self._validate_mixed
            print("Training with mixed precision")
        else:
            train_epoch = self._train_epoch
            validate = self._validate
            print("Training with single precision")

        progress = tqdm(total=iterations, desc=f"Epoch {self._epoch}", leave=True)
        msg = "Epoch %i: average [s/it]: %f, current metric: %f, best metric: %f"

        train_epochs = self.max_epoch - self._epoch
        for _ in range(train_epochs):
            t_per_iter = train_epoch(progress)
            current_metric = validate()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(current_metric)

            if current_metric < best_metric:
                best_metric = current_metric
                self._best_epoch = self._epoch
                self.save_checkpoint("best", best_metric)

            # TODO for tiny epochs we don"t want to save every time
            self.save_checkpoint("latest", best_metric)
            if self.early_stopping is not None:
                epochs_since_best = self._epoch - self._best_epoch
                if epochs_since_best > self.early_stopping:
                    print("Stopping training because there has been no improvement for", self.early_stopping, "epochs")
                    break

            self._epoch += 1
            progress.set_description(msg % (self._epoch, t_per_iter, current_metric, best_metric), refresh=True)

        print(f"Finished training after {self._epoch} epochs / {self._iteration} iterations.")
        print(f"The best epoch is number {self._best_epoch}.")

        if self._generate_name:
            self.name = None

        # TODO save the model to wandb if we have the wandb logger
        if isinstance(self.logger, WandbLogger):
            self.logger.get_wandb().finish()

    def _backprop(self, loss):
        loss.backward()
        self.optimizer.step()

    def _backprop_mixed(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _train_epoch(self, progress):
        return self._train_epoch_impl(progress, contextlib.nullcontext, self._backprop)

    def _train_epoch_mixed(self, progress):
        return self._train_epoch_impl(progress, amp.autocast, self._backprop_mixed)

    def _forward_and_loss(self, x, y):
        pred = self.model(x)
        if self._iteration % self.log_image_interval == 0:
            if pred.requires_grad:
                pred.retain_grad()

        loss = self.loss(pred, y)
        return pred, loss

    def _train_epoch_impl(self, progress, forward_context, backprop: Callable[[torch.Tensor], None]):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            with forward_context():
                pred, loss = self._forward_and_loss(x, y)

            backprop(loss)

            lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
            if self.logger is not None:
                self.logger.log_train(self._iteration, loss, lr, x, y, pred, log_gradients=True)

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _validate(self):
        return self._validate_impl(contextlib.nullcontext)

    def _validate_mixed(self):
        return self._validate_impl(amp.autocast)

    def _validate_impl(self, forward_context):
        self.model.eval()

        metric_val = 0.0
        loss_val = 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                with forward_context():
                    pred, loss = self._forward_and_loss(x, y)
                    metric = self.metric(pred, y)

                loss_val += loss.item()
                metric_val += metric.item()

        metric_val /= len(self.val_loader)
        loss_val /= len(self.val_loader)
        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric_val, loss_val, x, y, pred)
        return metric_val
