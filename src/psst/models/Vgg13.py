from torch import Tensor
from torch.nn import *


class Vgg13(Module):
    """VGG-13 architecture for processing 2D images.

    The VGG-13 name denotes that it contains 13 layers with trainable weights: 10 convolutional layers and 3 fully connected layers.

    This model expects an input of size (batch_size, 1, 224, 224)

    Reference:
    Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition.
    arXiv preprint arXiv:1409.1556.

    Usage:
        >>> model = Vgg13()
        >>> output = model(torch.randn(1, 1, 224, 224))

    Attributes:
        conv_stack (:class:`torch.nn.Sequential`): The sequence of layers in the VGG-13 network.
    """
    def __init__(self):
        super().__init__()

        self.conv_stack = Sequential(
            # Convolutional layers
            Conv2d(1, 64, 3, padding=1),  # 224, 224, 64
            ReLU(),
            Conv2d(64, 64, 3, padding=1),  # 224, 224, 64
            ReLU(),
            MaxPool2d(2),  # 112, 112, 64
            Conv2d(64, 128, 3, padding=1),  # 112, 112, 128
            ReLU(),
            Conv2d(128, 128, 3, padding=1),  # 112, 112, 128
            ReLU(),
            MaxPool2d(2),  # 56, 56, 128
            Conv2d(128, 256, 3, padding=1),  # 56, 56, 256
            ReLU(),
            Conv2d(256, 256, 3, padding=1),  # 56, 56, 256
            ReLU(),
            MaxPool2d(2),  # 28, 28, 256
            Conv2d(256, 512, 3, padding=1),  # 28, 28, 512
            ReLU(),
            Conv2d(512, 512, 3, padding=1),  # 28, 28, 512
            ReLU(),
            MaxPool2d(2),  # 14, 14, 512
            Conv2d(512, 512, 3, padding=1),  # 14, 14, 512
            ReLU(),
            Conv2d(512, 512, 3, padding=1),  # 14, 14, 512
            ReLU(),
            MaxPool2d(2),  # 7, 7, 512
            # Fully connected layers
            Flatten(),
            Linear(25088, 4096),
            ReLU(),
            Linear(4096, 4096),
            ReLU(),
            Linear(4096, 1000),
            ReLU(),
            Linear(1000, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the VGG-13 network.

        Args:
            x (:class:`torch.Tensor`): A 4D tensor with shape (batch_size, 1, 224, 224).

        Returns:
            :classL`torch.Tensor`: A 2D tensor with shape (batch_size, 1) representing the output of the network:
            A prediction of Bg or Bth.
        """
        return self.conv_stack(x)