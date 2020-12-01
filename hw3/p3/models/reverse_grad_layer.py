import torch
from torch.autograd import Function as AutoGrad_Function

class RevGrad(AutoGrad_Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = grad_output.neg() * ctx.alpha
        
        return grad_input, None

if __name__=='__main__':
    revgrad = RevGrad.apply

