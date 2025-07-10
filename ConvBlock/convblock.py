import torch
import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

# 卷积 ＋ BN ＋ RELU ＋ 卷积 ＋ BN；残差连接后再接relu
# 基本残差块结构，默认不改变空间尺寸和通道数
class BasicBlock(nn.Module):
    # 定义通道数的扩张系数
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        # 确保 BasicBlock 类正确继承并初始化了父类 nn.Module的属性和方法
        # py3中可以直接super().__init__()
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        # conv2：没有显式指定 stride，无论传入的stride是多少，其默认使用 stride=1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            # 如果downsample定义了，则用其调整残差的空间尺寸
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

# 卷积，BN，激活，卷积，BN，激活，卷积，BN，再加残差；
# 默认步长为1，不改变空间尺寸，默认通道扩张为2，会翻倍通道数
# 当使用bottleneck不指定步长为2时，空间尺寸保持不变
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

if __name__ == "__main__":
    # 设置输入张量的尺寸
    batch_size = 6
    inplanes = 64  # 输入通道数
    planes = 64    # 输出通道数
    H = 720        # 高度
    W = 960        # 宽度

    # 创建一个模拟的输入张量，尺寸为 (batch_size, inplanes, H, W)
    input_tensor = torch.randn(batch_size, inplanes, H, W)

    # 测试 BasicBlock，不改变通道数和空间尺寸（stride=1）
    basic_block = BasicBlock(inplanes=inplanes, planes=planes, stride=1, downsample=None, no_relu=False)
    # basic_block(input_tensor)后，底层会调用 BasciBlock 的 forward方法
    output_basic = basic_block(input_tensor)
    print(f"BasicBlock output shape (stride=1): {output_basic.shape}") # 输出：torch.Size([6, 64, 720, 960])

    # 测试 Bottleneck，需要考虑通道数的扩张（expansion=2）
    # 输出通道数为 planes * expansion，需要调整残差的通道数
    bottleneck_downsample = nn.Sequential(
        # 使用1x1的卷积来调整通道数
        nn.Conv2d(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=1, bias=False),
        BatchNorm2d(planes * Bottleneck.expansion, momentum=bn_mom),
    )
    bottleneck_block = Bottleneck(inplanes=inplanes, planes=planes, stride=1, downsample=bottleneck_downsample, no_relu=True)
    output_bottleneck = bottleneck_block(input_tensor)
    print(f"Bottleneck output shape (stride=1): {output_bottleneck.shape}") # 输出 torch.Size([6, 128, 720, 960])

    # 测试 BasicBlock，改变空间尺寸（stride=2）
    # 需要使用 downsample 调整残差的空间尺寸
    basic_block_downsample = nn.Sequential(
        # 使用1x1的卷积来调整空间尺寸，Bottleneck的expansion参数为2，但是BasicBlok的为1
        nn.Conv2d(inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=2, bias=False),
        BatchNorm2d(planes * BasicBlock.expansion, momentum=bn_mom),
    )
    basic_block_stride2 = BasicBlock(inplanes=inplanes, planes=planes, stride=2, downsample=basic_block_downsample, no_relu=False)
    output_basic_stride2 = basic_block_stride2(input_tensor)
    print(f"BasicBlock output shape (stride=2): {output_basic_stride2.shape}") # torch.Size([6, 64, 360, 480])

    # 测试 Bottleneck，改变空间尺寸（stride=2）
    # 需要使用 downsample 调整残差的空间尺寸和通道数
    bottleneck_downsample_stride2 = nn.Sequential(
        # 使用1x1的卷积来调整空间尺寸和通道数
        nn.Conv2d(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=2, bias=False),
        BatchNorm2d(planes * Bottleneck.expansion, momentum=bn_mom),
    )
    bottleneck_block_stride2 = Bottleneck(inplanes=inplanes, planes=planes, stride=2, downsample=bottleneck_downsample_stride2, no_relu=True)
    output_bottleneck_stride2 = bottleneck_block_stride2(input_tensor)
    print(f"Bottleneck output shape (stride=2): {output_bottleneck_stride2.shape}") # torch.Size([6, 128, 360, 480])