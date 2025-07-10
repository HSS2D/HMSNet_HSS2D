import torch
from torch import nn
from torch.nn import functional as F

# 计算 Dice 损失，用于衡量预测结果与真实标签之间的相似性
def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


# DetailAggregateLoss 结合边界信息的 BCE 损失和 Dice 损失，提升模型在语义分割任务中的细节和边界识别能力。
class DetailAggregateLoss(nn.Module):
    def __init__(self):
        super(DetailAggregateLoss, self).__init__()
        # 定义拉普拉斯核，用于边缘检测。
        self.laplacian_kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
        # 定义可学习参数的融合核，用于加权融合多尺度的边界信息。
        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6. / 10], [3. / 10], [1. / 10]], dtype=torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks):
        # 边界目标提取
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        # 多尺度边界提取
        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)

        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)

        # 上采样回原始尺度
        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        # 二值化
        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0

        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0

        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0

        # 多尺度边界堆叠
        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)

        # 融合多尺度边界信息
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)
        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0

        # 调整预测边界尺寸
        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)

        # 计算损失
        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        return bce_loss, dice_loss


# OHEM 是一种策略，用于在训练过程中动态选择那些当前模型表现较差的样本（即“困难样本”）来计算损失和进行梯度更新。
# 这样可以使模型更关注难以分类的样本，提升其在复杂情况下的泛化能力。
class OhemCELoss(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7, min_kept=65535, weight=None):
        super(OhemCELoss, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _ohem_forward(self, score, target):
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()

        try:
            min_value = pred[min(self.min_kept, pred.numel() - 1)]
            threshold = max(min_value, self.thresh)
        except IndexError:
            try:
                print("使用默认的阈值")
                threshold = self.thresh
            except:
                print("===== ! ATTENTION ERROR OCCURRED ! =====")
        # min_value = pred[min(self.min_kept, pred.numel() - 1)]
        # threshold = max(min_value, self.thresh)
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return torch.mean(pixel_losses)

    def forward(self, score, target):
        return self._ohem_forward(score, target)