import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def gen_conv(x,
             cnum,
             ksize,
             stride=1,
             rate=1,
             name='conv',
             padding='SAME',
             activation=nn.ELU,
             training=True):
    """Define conv for generator.
    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.
    Returns:
        paddle.Tensor: output
    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFLECT']
    if padding == 'SYMMETRIC' or padding == 'REFLECT':
        p = int(rate * (ksize - 1) / 2)
        x = paddle.nn.functional.pad(x, [p, p, p, p], mode=padding.lower())
        padding = 'VALID'

    conv = nn.Conv2D(in_channels=int(x.shape[1]),
                     out_channels=cnum,
                     kernel_size=ksize,
                     stride=stride,
                     padding=0 if padding == 'VALID' else (ksize - 1) // 2,
                     dilation=rate,
                     weight_attr=paddle.ParamAttr(name=name + "_w"),
                     bias_attr=paddle.ParamAttr(name=name + "_b"))

    x = conv(x)

    if cnum == 3 or activation is None:
        # conv for output
        return x

    x, y = paddle.split(x, num_or_sections=2, axis=1)
    x = activation()(x)
    y = paddle.nn.functional.sigmoid(y)
    x = x * y
    return x


def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.
    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.
    Returns:
        paddle.Tensor: output
    """
    with paddle.static.name_scope(name):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = gen_conv(x,
                     cnum,
                     3,
                     1,
                     name=name + '_conv',
                     padding=padding,
                     training=training)
    return x