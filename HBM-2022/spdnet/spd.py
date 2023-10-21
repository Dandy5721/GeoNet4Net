import torch
from torch import nn
from spdnet import StiefelParameter


class SPDTransform(nn.Module):
    def __init__(self, input_size, output_size, in_channels=1):
        super(SPDTransform, self).__init__()

        if in_channels > 1:
            self.weight = StiefelParameter(
                torch.Tensor(in_channels, input_size, output_size), requires_grad=True
            )
        else:
            self.weight = StiefelParameter(
                torch.Tensor(input_size, output_size), requires_grad=True
            )
        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        weight = self.weight
        output = weight.transpose(-2, -1) @ input @ weight
        return output


class SPDTangentSpace(nn.Module):
    def __init__(self):
        super(SPDTangentSpace, self).__init__()

    def forward(self, input):
        s, u = torch.linalg.eigh(input)
        s = s.log().diag_embed()
        output = u @ s @ u.transpose(-2, -1)
        return torch.flatten(output, 1)


class SPDExpMap(nn.Module):
    def __init__(self):
        super(SPDExpMap, self).__init__()

    def forward(self, input):
        s, u = torch.linalg.eigh(input)
        s = s.exp().diag_embed()
        output = u @ s @ u.transpose(-2, -1)
        return output


class SPDRectified(nn.Module):
    def __init__(self, epsilon=1e-4):
        super(SPDRectified, self).__init__()
        self.register_buffer('epsilon', torch.DoubleTensor([epsilon]))

    def forward(self, input):
        s, u = torch.linalg.eigh(input)
        s = s.clamp(min=self.epsilon[0])
        s = s.diag_embed()
        output = u @ s @ u.transpose(-2, -1)

        return output
