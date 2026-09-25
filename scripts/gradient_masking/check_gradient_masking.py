import numpy as np
import torch
import torch_em


class ApplyMask_new:
    def __call__(self, prediction, target, mask):
        mask.requires_grad = False
        mask = mask.type(torch.bool)
        # prediction = prediction * mask
        # target = target * mask
        # FIXME: I am not sure this will not do the right thing if we have channels,
        # since we want to preserve the channel axis
        prediction = prediction[mask][None][None]  # add aditional C and N axis for flatten_samples(input_)
        target = target[mask][None][None]
        return prediction, target


def compare_gradients(gradients, mask):

    # check that the gradients are zero outside of the mask
    print("Gradients are zero outside of mask:")
    for name, grad in gradients.items():
        print(name, np.allclose(grad[~mask], 0))

    # check overall agreement of gradients
    print()
    print("Gradient agreement:")
    agreements = np.zeros((len(gradients), len(gradients)))
    for i, (name_a, grad_a) in enumerate(gradients.items()):
        for j, (name_b, grad_b) in enumerate(gradients.items()):
            # if i > j:
            #     continue
            assert grad_a.shape == grad_b.shape
            agree = float(np.isclose(grad_a, grad_b).sum()) / grad_a.size
            agreements[i, j] = agree

    # TODO display this like a confusion matrix
    print(agreements)


def check_gradient_masking(loss_function):
    torch.manual_seed(42)

    shape = (1, 1, 64, 64)
    pred = torch.rand(*shape)
    pred.requires_grad = True

    target = torch.randint(0, 2, shape).type(torch.float)

    mask = torch.zeros(*shape)
    mask[:, :, 10:20, 15:25] = 1

    # compute loss without masking
    loss_normal = loss_function(pred, target)
    loss_normal.backward()
    grad_normal = pred.grad

    # compute cropped loss
    pred.grad = None
    loss_cropped = loss_function(pred[:, :, 10:20, 15:25], target[:, :, 10:20, 15:25])
    loss_cropped.backward()
    grad_cropped = pred.grad

    # compute masked loss (torch-em impl)
    pred.grad = None
    wrapper = torch_em.loss.LossWrapper(loss_function, torch_em.loss.ApplyMask())
    loss_masked1 = wrapper(pred, target, mask=mask)
    loss_masked1.backward()
    grad_masked1 = pred.grad

    # compute masked loss (new impl)
    pred.grad = None
    wrapper = torch_em.loss.LossWrapper(loss_function, ApplyMask_new())
    loss_masked2 = wrapper(pred, target, mask=mask)
    loss_masked2.backward()
    grad_masked2 = pred.grad

    # compare the gradients
    gradients = {
        "grad_normal": grad_normal.detach().numpy(),
        "grad_cropped": grad_cropped.detach().numpy(),
        "grad_masked1": grad_masked1.detach().numpy(),
        "grad_masked2": grad_masked2.detach().numpy(),
    }
    compare_gradients(gradients, mask=mask.detach().numpy().astype("bool"))


def main():
    print("Check Dice loss")
    check_gradient_masking(torch_em.loss.DiceLoss())

    print()
    print()
    print("Check MSE loss")
    check_gradient_masking(torch.nn.MSELoss())


if __name__ == "__main__":
    main()
