import torch

class Vgg13(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_stack = torch.nn.Sequential(
            # Convolutional layers
            torch.nn.Conv2d(1, 64, 3, padding=1),       # 224, 224, 64
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),      # 224, 224, 64
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                      # 112, 112, 64
            torch.nn.Conv2d(64, 128, 3, padding=1),     # 112, 112, 128
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),    # 112, 112, 128
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                      # 56, 56, 128
            torch.nn.Conv2d(128, 256, 3, padding=1),    # 56, 56, 256
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),     # 56, 56, 256
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                      # 28, 28, 256
            torch.nn.Conv2d(256, 512, 3, padding=1),    # 28, 28, 512
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),    # 28, 28, 512
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                      # 14, 14, 512
            torch.nn.Conv2d(512, 512, 3, padding=1),    # 14, 14, 512
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),    # 14, 14, 512
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                      # 7, 7, 512

            # Fully connected layers
            torch.nn.Flatten(),
            torch.nn.Linear(25088,4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096,4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096,1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 1)
        )

    def forward(self, x):
        return self.conv_stack(x)
