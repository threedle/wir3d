import torch

class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    # @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    # @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None

def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)

# Approximates cubic bezier length using piecewise linear approximation
def bcurve_length(points, num_samples=1000):
    """
        points: B x 4 x 3 (torch.tensor) batched or unbatched
        num_samples: int
    """
    # Bezier weighting
    sample_t = torch.linspace(0, 1, num_samples).to(points.device)

    # NOTE: bezier interpolation follows binomial distribution so the weights will always sum to 1
    w0 = (1 - sample_t)**3
    w1 = 3 * (1 - sample_t)**2 * sample_t
    w2 = 3 * (1 - sample_t) * sample_t**2
    w3 = sample_t**3

    if len(points.shape) == 2:
        # Unbatched
        samples = w0[:, None] * points[None, 0] + w1[:, None] * points[None, 1] + \
                    w2[:, None] * points[None, 2] + w3[:, None] * points[None, 3] # N x 3
        length = torch.linalg.norm(samples[1:] - samples[:-1], dim=1).sum()
    else:
        samples = w0.reshape(1, -1, 1) * points[:, None, 0] + \
                    w1.reshape(1, -1, 1) * points[:, None, 1] + \
                    w2.reshape(1, -1, 1) * points[:, None, 2] + \
                        w3.reshape(1, -1, 1) * points[:, None, 3]
        length = torch.linalg.norm(samples[:, 1:] - samples[:, :-1], dim=2).sum(dim=1)

    return length