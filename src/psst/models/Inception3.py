from torch import Tensor, cat
from torch.nn import *


class BasicConv2d(Module):
    """Custom Conv2d Module with :code:`bias=False` by default.
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()

        self.conv = Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return functional.relu(x, inplace=True)


class Conv2dTo2Channels(Module):
    """Custom Module that creates two channels, one each from a pass of a 
    :class:`BasicConv2d` with orthogonal, asymmetric kernels. Equivalent to:

    Example:
        This code:
        >>> x = torch.rand((8, 8))
        >>> model = Conv2dTo2Channels(4, 4, kernel_size=(1, 3), padding=(0, 1))
        >>> y = model(x)

        should be equivalent to:
        >>> x = torch.rand((8, 8))
        >>> model1 = BasicConv2d(4, 4, kernel_size=(1, 3), padding=(0, 1))
        >>> model2 = BasicConv2d(4, 4, kernel_size=(3, 1), padding=(1, 0))
        >>> y = torch.cat((model1(x), model2(x)), 1)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int],
        padding: tuple[int, int]
    ):
        super().__init__()

        self.channel1 = BasicConv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.channel2 = BasicConv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size[1], kernel_size[0]),
            padding=(padding[1], padding[0]),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cat((self.channel1(x), self.channel2(x)), 1)


class InceptionA(Module):
    """An inception block containing four parallel branches: 
    1. 1x1 convolution
    2. 1x1 convolution -> 5x5 convolution
    3. 1x1 convolution -> 3x3 convolution -> 3x3 convolution
    4. Average pooling -> 1x1 convolution

    Usage:
        >>> block = InceptionA(in_channels = 192, pool_features = 32)

    Args:
        >>> in_channels (int): Number of input channels.
        >>> pool_features (int): Number of output channels for the pooling branch.

    """
    def __init__(self, in_channels: int, pool_features: int):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5 = Sequential(
            BasicConv2d(in_channels, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5, padding=2),
        )
        self.branch3x3dbl = Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1),
        )
        self.branch_pool = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_features, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the InceptionA block
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: Output tensor after passing through InceptionA block.
        """
        return cat(
            (
                self.branch1x1(x),
                self.branch5x5(x),
                self.branch3x3dbl(x),
                self.branch_pool(x),
            ),
            1,
        )


class InceptionB(Module):
    """An inception block containing three parallel branches: 
    1. 3x3 convolution
    2. 1x1 convolution -> 3x3 convolution -> 3x3 convolution
    3. Average pooling

    Usage:
        >>> block = InceptionB(in_channels = 192)

    Args:
        >>> in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels: int):
        super().__init__()

        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl = Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=2),
        )
        self.branch_pool = AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x: Tensor):
        """Forward pass of the InceptionB block
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: Output tensor after passing through InceptionB block.
        """
        return cat((self.branch3x3(x), self.branch3x3dbl(x), self.branch_pool(x)), 1)


class InceptionC(Module):
    """An inception block containing four parallel branches:
    1. 1x1 convolution
    2. 1x1 convolution -> 1x7 convolution -> 7x1 convolution
    3. 1x1 convolution -> 7x1 convolution -> 1x7 convolution -> 7x1 convolution -> 1x7 convolution
    4. Average pooling -> 1x1 convolution

    Usage:
        >>> block = InceptionC(in_channels = 192, channels_7x7 = 128)

    Args:
        >>> in_channels (int): Number of input channels.
        >>> channels_7x7 (int): Number of channels for the 7x7 convolution.

    """
    def __init__(self, in_channels: int, channels_7x7: int):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7 = Sequential(
            BasicConv2d(in_channels, channels_7x7, kernel_size=1),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(channels_7x7, 192, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.branch7x7dbl = Sequential(
            BasicConv2d(in_channels, channels_7x7, kernel_size=1),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(channels_7x7, 192, kernel_size=(1, 7), padding=(0, 3)),
        )
        self.branch_pool = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the InceptionC block
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: Output tensor after passing through InceptionC block.
        """
        return cat(
            (
                self.branch1x1(x),
                self.branch7x7(x),
                self.branch7x7dbl(x),
                self.branch_pool(x),
            ),
            1,
        )


class InceptionD(Module):
    """An inception block containing three parallel branches:
    1. 1x1 convolution -> 3x3 convolution
    2. 1x1 convolution -> 1x7 convolution -> 7x1 convolution -> 3x3 convolution
    3. Average pooling

    Usage:
        >>> block = InceptionD(in_channels = 192, pool_features = 32)

    Args:
        >>> in_channels (int): Number of input channels.
        >>> pool_features (int): Number of output channels for the pooling branch.

    """
    def __init__(self, in_channels: int):
        super().__init__()

        self.branch3x3 = Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 320, kernel_size=3, stride=2),
        )
        self.branch7x7x3 = Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(192, 192, kernel_size=3, stride=2),
        )
        self.branch_pool = AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the InceptionD block

        :param x: Input Tensor
        :type x: Tensor
        :returns: Concatenated output for all branches
        :rtype: Tensor
        """
        return cat((self.branch3x3(x), self.branch7x7x3(x), self.branch_pool(x)), 1)


class InceptionE(Module):
    """An inception block containing four parallel branches, with two branches creating two parallel branches each:
    1. 1x1 convolution
    2. 1x1 convolution -> 
        2a. 1x3 convolution
        2b. 3x1 convolution
    3. 1x1 convolution -> 3x3 convolution ->
        3a. 1x3 convolution
        3b. 3x1 convolution
    4. Average pooling -> 1x1 convolution

    Usage:
        >>> block = InceptionE(in_channels = 192)

    Args:
        >>> in_channels (int): Number of input channels.

    """
    def __init__(self, in_channels: int):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3 = Sequential(
            BasicConv2d(in_channels, 384, kernel_size=1),
            Conv2dTo2Channels(384, 384, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.branch3x3dbl = Sequential(
            BasicConv2d(in_channels, 448, kernel_size=1),
            BasicConv2d(448, 384, kernel_size=3, padding=1),
            Conv2dTo2Channels(384, 384, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.branch_pool = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the InceptionE block

        :param x: Input Tensor
        :type x: Tensor
        :returns: Concatenated output for all branches
        :rtype: Tensor
        """
        return cat(
            (
                self.branch1x1(x),
                self.branch3x3(x),
                self.branch3x3dbl(x),
                self.branch_pool(x),
            ),
            1,
        )


class Inception3(Module):
    """Inception-block-based neural network for training on 2D images.

    This is a modified inceptionv3-block-based model, which utilizes 
    several inception blocks for processing 2D image data. The model
    expects input tensors to be 3D or 4D and automatically adjusts the
    3D inputs.

    Usage:
        >>> model = Inception3()
        >>> output = model(torch.randn(1, 192, 28, 28))

    Attributes:
        stack (:class:`torch.nn.Sequential`): A sequential stack of layers
        including convolutional, pooling, and inception blocks.
    """
    def __init__(self):
        """Initializes the modified Inception v3 model with predefined layer structure.
        """
        super().__init__()

        self.stack = Sequential(
            BasicConv2d(1, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            MaxPool2d(kernel_size=3, stride=2),
            BasicConv2d(64, 80, kernel_size=1),
            BasicConv2d(80, 192, kernel_size=3),
            MaxPool2d(kernel_size=3, stride=2),
            InceptionA(192, pool_features=32),
            InceptionA(256, pool_features=64),
            InceptionA(288, pool_features=64),
            InceptionB(288),
            InceptionC(768, channels_7x7=128),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=192),
            InceptionD(768),
            InceptionE(1280),
            InceptionE(2048),
            AdaptiveAvgPool2d((1, 1)),
            Flatten(1),
            Linear(2048, 1),
            Flatten(0),
        )

    def _transform_input(self, x: Tensor) -> Tensor:
        """Transforms the input tensor to match the expected 4D shape for the model
        
        Args:
            x (:class:`torch.Tensor`): Input tensor. Expected to be 3D or 4D.

        Returns:
            :class:`torch.Tensor`: 4D transformed input tensor.
        """
        if x.ndim == 3:
            x.unsqueeze_(1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Defines the forward pass of the model.
        
        Args:
            x (:class:`torch.Tensor`): Input tensor, expected to be 3D or 4D.

        Returns:
            :class:`torch.Tensor`: The output tensor after the forward pass through
            the model's layers.
        """
        x = self._transform_input(x)
        return self.stack(x)
