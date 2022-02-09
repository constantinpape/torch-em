from .util import get_trainer


def _get_n_images(loader):
    ds = loader.dataset
    n_images = None
    if "ImageCollectionDataset" in str(ds):
        n_images = len(ds.raw_images)
    # TODO cover other cases
    return n_images


def _get_training_summary(trainer, lr):

    n_epochs = trainer.epoch
    batches_per_epoch = len(trainer.train_loader)
    batch_size = trainer.train_loader.batch_size
    print("The model was trained for", n_epochs, "epochs with length", batches_per_epoch, "and batch size", batch_size)

    loss = str(trainer.loss)
    if loss.startswith("LossWrapper"):
        loss = loss.split("\n")[1]
        index = loss.find(":")
        loss = loss[index+1:]
    loss = loss.replace(" ", "").replace(")", "").replace("(", "")
    print("It was trained with", loss, "as loss function")

    opt_ = str(trainer.optimizer)
    if lr is None:
        print("Learning rate is determined from optimizer - this will be the final, not initial learning rate")
        i0 = opt_.find("lr:")
        i1 = opt_.find("\n", i0)
        lr = opt_[i0+3:i1].replace(" ", "")
    opt = opt_[:opt_.find(" ")]
    print("And using the", opt, "optimizer with learning rate", lr)

    n_train = _get_n_images(trainer.train_loader)
    n_val = _get_n_images(trainer.val_loader)
    print(n_train, "images were used for training and", n_val, "for validation")

    report = dict(
        n_epochs=n_epochs, batches_per_epoch=batches_per_epoch, batch_size=batch_size,
        loss_function=loss, optimizer=opt, learning_rate=lr,
        n_train_images=n_train, n_validation_images=n_val
    )
    if n_train is not None:
        report["n_train_images"] = n_train
    if n_val is not None:
        report["n_val_images"] = n_val
    return report


def get_training_summary(
    ckpt, lr=None, model_name="best", to_md=False
):
    trainer = get_trainer(ckpt, name=model_name)
    print("Model summary for", ckpt, "using the", model_name, "model")
    training_summary = _get_training_summary(trainer, lr)
    if to_md:
        training_summary = "\n".join(f"- {k}: {v}" for k, v in training_summary.items())
    return training_summary
