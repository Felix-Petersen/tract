# Copyright (c) Felix Petersen.  https://github.com/Felix-Petersen/tract
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch


class TrActLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, lambda_, inv_type):
        ctx.save_for_backward(input, weight, bias)
        ctx.lambda_ = lambda_
        if inv_type == 'cholesky_inverse':
            ctx.inverse = torch.cholesky_inverse
        elif inv_type == 'inverse':
            ctx.inverse = torch.inverse
        else:
            raise NotImplementedError(inv_type)
        return input @ weight.T + (bias if bias is not None else 0)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        assert len(input.shape) == 2, input.shape
        assert len(grad_output.shape) == 2, grad_output.shape

        if ctx.needs_input_grad[0]:
            grad_0 = grad_output @ weight
        else:
            grad_0 = None

        if ctx.needs_input_grad[1]:
            if input.shape[0] < input.shape[1]:
                aaT = input @ input.T / input.shape[0]
                I_b = torch.eye(aaT.shape[0], device=aaT.device, dtype=aaT.dtype)
                aaT_IaaT_inv = aaT @ ctx.inverse(aaT / ctx.lambda_ + I_b)
                grad_1 = grad_output.T @ (
                        I_b - 1. / ctx.lambda_ * aaT_IaaT_inv
                ) @ input / ctx.lambda_

            else:
                aTa = input.T @ input / input.shape[0]
                I_n = torch.eye(aTa.shape[0], device=aTa.device, dtype=aTa.dtype)
                grad_1 = grad_output.T @ input @ ctx.inverse(aTa + ctx.lambda_ * I_n)

        else:
            grad_1 = None

        return (
            grad_0,
            grad_1,
            grad_output.sum(0, keepdim=True) if bias is not None else None,
            None, None
        )


class TrActConv2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, lambda_, inv_type):
        ctx.save_for_backward(input, weight, bias)
        ctx.lambda_ = lambda_
        ctx.stride = stride
        ctx.padding = padding
        if inv_type == 'cholesky_inverse':
            ctx.inverse = torch.cholesky_inverse
        elif inv_type == 'inverse':
            ctx.inverse = torch.inverse
        else:
            raise NotImplementedError(inv_type)
        return torch.nn.functional.conv2d(input, weight, bias, stride, padding)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        assert len(input.shape) == 4, input.shape
        assert len(grad_output.shape) == 4, grad_output.shape

        if ctx.needs_input_grad[0]:
            grad_0 = torch.nn.grad.conv2d_input(
                input.shape, weight, grad_output, stride=ctx.stride, padding=ctx.padding
            )
        else:
            grad_0 = None

        if ctx.needs_input_grad[1]:
            unf = torch.nn.functional.unfold(
                input=input, kernel_size=(weight.shape[2], weight.shape[3]), padding=ctx.padding, stride=ctx.stride
            )
            a = unf.transpose(1, 2).reshape(unf.shape[0] * unf.shape[2], unf.shape[1])
            assert list(a.shape) == [
                input.shape[0] * grad_output.shape[2] * grad_output.shape[3],
                weight.shape[1] * weight.shape[2] * weight.shape[3]
            ], (
                a.shape,
                input.shape[0] * grad_output.shape[2] * grad_output.shape[3],
                weight.shape[1] * weight.shape[2] * weight.shape[3]
            )
            if input.shape[0] * grad_output.shape[2] * grad_output.shape[3] < weight.shape[1] * weight.shape[2] * weight.shape[3]:
                aaT = a @ a.T / a.shape[0]
                I_b = torch.eye(aaT.shape[0], device=aaT.device, dtype=aaT.dtype)
                aaT_IaaT_inv = aaT @ ctx.inverse(aaT / ctx.lambda_ + I_b)
                grad_1 = (grad_output.transpose(0, 1).reshape(grad_output.shape[1], -1) @ (
                        I_b - 1. / ctx.lambda_ * aaT_IaaT_inv
                ) @ a).reshape(*weight.shape) / ctx.lambda_
            else:
                aTa = a.T @ a / a.shape[0]
                I_n = torch.eye(aTa.shape[0], device=aTa.device, dtype=aTa.dtype)
                grad_weight = torch.nn.grad.conv2d_weight(
                    input, weight.shape, grad_output, stride=ctx.stride, padding=ctx.padding
                )
                grad_1 = (
                        grad_weight.reshape(grad_weight.shape[0], -1) @ ctx.inverse(aTa + ctx.lambda_ * I_n)
                ).reshape(*grad_weight.shape)
        else:
            grad_1 = None

        return (
            grad_0,
            grad_1,
            grad_output.sum(0).sum(-1).sum(-1) if bias is not None else None,
            None, None, None, None
        )


class TrActLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features,
                 lambda_, inv_type='inverse', **kwargs):
        super(TrActLinear, self).__init__(
            in_features=in_features, out_features=out_features, **kwargs
        )
        self.lambda_ = lambda_
        self.inv_type = inv_type

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_shape = input.shape
        if len(input_shape) > 2:
            input = input.view(-1, input.shape[-1])

        output = TrActLinearFunction.apply(
            input, self.weight,
            self.bias.unsqueeze(0) if self.bias is not None else None,
            self.lambda_,
            self.inv_type
        )

        if len(input_shape) > 2:
            output = output.view(*input_shape[:-1], output.shape[-1])

        return output


class TrActConv2d(torch.nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            lambda_=None,
            inv_type='inverse',
            **kwargs
    ):
        assert lambda_ is not None
        super(TrActConv2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, **kwargs
        )
        self.lambda_ = lambda_
        self.inv_type = inv_type

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return TrActConv2DFunction.apply(
            input, self.weight,
            self.bias,
            self.stride, self.padding,
            self.lambda_,
            self.inv_type
        )


def TrAct(self, l=0.1):
    assert isinstance(self, torch.nn.Linear) or isinstance(self, torch.nn.Conv2d), (
        'The input needs to be a torch.nn.Linear or a torch.nn.Conv2d module. `{}`'.format(self)
    )
    if isinstance(self, torch.nn.Linear):
        new_self = TrActLinear(self.in_features, self.out_features, lambda_=l)

    elif isinstance(self, torch.nn.Conv2d):
        new_self = TrActConv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
            lambda_=l
        )

    new_self.weight = self.weight
    new_self.bias = self.bias
    return new_self


if __name__ == '__main__':
    exit()  # comment out line for some tests
    torch.manual_seed(0)

    x = torch.randn(30, 8)
    y = torch.randn(30, 5)
    lin = TrActLinear(8, 5, 0.1)
    y_hat = lin(x)
    loss = (y - y_hat).pow(2).mean()
    loss.backward()

    print(lin.weight.grad)
    print(lin.bias.grad)

    """
tensor([[ 0.0741,  0.0949, -0.0559, -0.0398, -0.1364,  0.0013,  0.1603, -0.0468],
        [-0.0994, -0.1061, -0.1781,  0.1479,  0.1114, -0.0374,  0.0324, -0.1329],
        [ 0.0303,  0.0484,  0.1385, -0.1675,  0.0778, -0.0854, -0.0592,  0.0526],
        [-0.0726,  0.1049,  0.0998,  0.0420, -0.1409, -0.0031, -0.0432, -0.1150],
        [ 0.1438, -0.0729, -0.0261, -0.1015, -0.0902,  0.1821, -0.1520,  0.0950]])
tensor([-0.1162,  0.1062, -0.1275, -0.0206, -0.1149])
    """

    torch.manual_seed(0)

    x = torch.randn(30, 2, 6, 4)
    y = torch.randn(30, 5, 4, 2)
    con = TrActConv2d(2, 5, 3, 1, 0, 0.1)
    y_hat = con(x)
    loss = (y - y_hat).pow(2).mean()
    loss.backward()

    print(con.weight.grad)
    print(con.bias.grad)

    """
tensor([[[[-0.0501, -0.0166,  0.0214],
          [-0.0230,  0.0324,  0.0655],
          [ 0.0010,  0.0600,  0.0290]],

         [[-0.0144,  0.0254,  0.0484],
          [ 0.0741, -0.0920,  0.0260],
          [-0.0109, -0.0480, -0.0420]]],


        [[[ 0.0408,  0.0736, -0.0403],
          [ 0.0959,  0.0447,  0.0549],
          [ 0.0449,  0.0744, -0.0351]],

         [[-0.0344, -0.0557,  0.0050],
          [ 0.0286, -0.0264,  0.0241],
          [ 0.0068, -0.0687,  0.0100]]],


        [[[ 0.0583, -0.0044, -0.0171],
          [ 0.0665, -0.0569, -0.0228],
          [-0.0463,  0.0894, -0.0777]],

         [[-0.0105, -0.0507,  0.0191],
          [ 0.0222,  0.0361,  0.0910],
          [ 0.0371, -0.0148,  0.0335]]],


        [[[ 0.0478,  0.0064,  0.0882],
          [-0.0615,  0.0065, -0.0027],
          [ 0.0335, -0.0021,  0.0069]],

         [[-0.0405, -0.0451, -0.0285],
          [ 0.0239,  0.0784,  0.0748],
          [ 0.0436, -0.0527, -0.0703]]],


        [[[-0.0896,  0.0370, -0.0823],
          [-0.0971, -0.0723, -0.0460],
          [ 0.0048, -0.0286, -0.0452]],

         [[ 0.0912, -0.0151,  0.0588],
          [-0.0053, -0.0359, -0.0247],
          [-0.0263, -0.0650,  0.0599]]]])
tensor([ 0.0052, -0.0397,  0.1093,  0.0464, -0.1079])
    """

    ####################################################################################################################
    #  Runtime Testing  ################################################################################################
    ####################################################################################################################
    exit()  # comment out line for runtime testing plots

    import matplotlib.pyplot as plt
    import time
    import tqdm

    bss = list(range(1, 101, 1))
    num_it = 10

    times = []
    con = TrActConv2d(50, 1024, 6, 1, 0, 0.1)
    for i in tqdm.tqdm(bss):
        x = torch.randn(i, 50, 12, 12)
        t_s = time.time()
        for _ in range(num_it):
            y_hat = con(x)
        times.append((time.time() - t_s) * 1000 / num_it)

    plt.plot(bss, times, label='Conv2D')

    times = []
    con = TrActConv2d(50, 1024, 6, 1, 0, 0.1)
    for i in tqdm.tqdm(bss):
        x = torch.randn(i, 50, 12, 12)
        y = torch.randn(i, 1024, 7, 7)
        t_s = time.time()
        for _ in range(num_it):
            y_hat = con(x)
            loss = (y - y_hat).pow(2).mean()
            loss.backward()
        times.append((time.time() - t_s) * 1000 / num_it)

    plt.plot(bss, times, label='Conv2D with backward')
    plt.legend()
    plt.ylabel('[ms]')
    plt.xlabel('bs')
    plt.show()

