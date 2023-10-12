import collections
from typing import Optional
from warnings import warn

import torch

InceptionOutputs = collections.namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': torch.Tensor, 'aux_logits': Optional[torch.Tensor]}

class BasicConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return torch.nn.functional.relu(x, inplace=True)

class InceptionA(torch.nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size = 1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size = 3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size = 3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionB(torch.nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size = 1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size = 3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size = 3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionC(torch.nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1,7), padding=(0,3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7,1), padding=(3,0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size = 1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size =(7,1), padding=(3,0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size =(1,7), padding=(0,3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size =(7,1), padding=(3,0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size =(1,7), padding=(0,3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionD(torch.nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1,7), padding=(0,3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7,1), padding=(3,0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionE(torch.nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1,3), padding=(0,1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3,1), padding=(1,0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size = 1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size =(1,3), padding=(0,1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size =(3,1), padding=(1,0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionAux(torch.nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = torch.nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Inception3(torch.nn.Module):

    def __init__(self, aux_logits=False, transform_input=False):
        super().__init__()
        
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.Conv2d_1a_3x3: torch.nn.Module = BasicConv2d(1, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3: torch.nn.Module = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3: torch.nn.Module = BasicConv2d(32, 64, kernel_size=3, padding=1)
        
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.Conv2d_3b_1x1: torch.nn.Module = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3: torch.nn.Module = BasicConv2d(80, 192, kernel_size=3)
        
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.Mixed_5b: torch.nn.Module = InceptionA(192, pool_features=32)
        self.Mixed_5c: torch.nn.Module = InceptionA(256, pool_features=64)
        self.Mixed_5d: torch.nn.Module = InceptionA(288, pool_features=64)
        self.Mixed_6a: torch.nn.Module = InceptionB(288)
        self.Mixed_6b: torch.nn.Module = InceptionC(768, channels_7x7=128)
        self.Mixed_6c: torch.nn.Module = InceptionC(768, channels_7x7=160)
        self.Mixed_6d: torch.nn.Module = InceptionC(768, channels_7x7=160)
        self.Mixed_6e: torch.nn.Module = InceptionC(768, channels_7x7=192)

        if aux_logits:
            self.AuxLogits: torch.nn.Module = InceptionAux(768, 1)
        
        self.Mixed_7a: torch.nn.Module = InceptionD(768)
        self.Mixed_7b: torch.nn.Module = InceptionE(1280)
        self.Mixed_7c: torch.nn.Module = InceptionE(2048)
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(2048, 1)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 1:
            x.unsqueeze_(1)
        return x

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)

        x = self.maxpool1(x)
        
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        
        x = self.maxpool2(x)
        
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux: Optional[torch.Tensor] = self.AuxLogits(x)
        else:
            aux = None

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x.flatten(), aux

    @torch.jit.unused
    def eager_outputs(self, x: torch.Tensor, aux: torch.Tensor):
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x

    def forward(self, x: torch.Tensor):
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)
