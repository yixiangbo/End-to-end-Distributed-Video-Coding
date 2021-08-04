import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.modules.utils import _pair


class ConvRNNCellBase(nn.Module):
    def __repr__(self):
        s = (
            '{name}({input_channels}, {hidden_channels}, kernel_size={kernel_size}'
            ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        s += ', hidden_kernel_size={hidden_kernel_size}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ConvLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 input_channels,  # 输入通道
                 hidden_channels,  # 输出通道
                 kernel_size=3,  # 卷积核大小
                 stride=1,  # 卷积步长
                 padding=0,
                 dilation=1,
                 hidden_kernel_size=1,
                 bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.hidden_kernel_size = _pair(hidden_kernel_size)

        hidden_padding = _pair(hidden_kernel_size // 2)

        gate_channels = 4 * self.hidden_channels  # 隐藏通道数乘以4
        self.conv_ih = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=gate_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias)

        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=gate_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            dilation=1,
            bias=bias)

        """
        参数默认初始化 https://blog.csdn.net/qq_36338754/article/details/97756378
        """
        self.reset_parameters()

    def reset_parameters(self):
        self.conv_ih.reset_parameters()
        self.conv_hh.reset_parameters()

    def forward(self, input, hidden):
        hx, cx = hidden

        gates = self.conv_ih(input) + self.conv_hh(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)  # 由通道数分割为四份
        """
        https://zhuanlan.zhihu.com/p/148122328
        对照 2.2公式
        """
        ingate = torch.sigmoid(ingate)  # input gate
        forgetgate = torch.sigmoid(forgetgate)  # forget gate
        cellgate = torch.tanh(cellgate)  # New memory cell
        outgate = torch.sigmoid(outgate)  # output gate

        cy = (forgetgate * cx) + (ingate * cellgate)  # Final memory cell
        hy = outgate * torch.tanh(cy)

        return hy, cy
