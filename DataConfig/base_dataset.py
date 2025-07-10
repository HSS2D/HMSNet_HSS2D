import cv2
import numpy as np
import random

from torch.nn import functional as F
from torch.utils import data

y_k_size = 6
x_k_size = 6

# BaseDataset 是一个自定义的 PyTorch 数据集类，继承自 torch.utils.data.Dataset，用于加载、处理和增强图像数据
class BaseDataset(data.Dataset):
    def __init__(self,
                 # 用于标签图像中忽略的标签值，通常是背景或无效区域的标签。
                 ignore_label=255,
                 # 图像基准尺寸
                 base_size=2048,
                 # 训练时，裁剪的目标尺寸
                 crop_size=(512, 1024),
                 # 随机缩放时的比例因子范围
                 scale_factor=16,
                 # 标准化图像的均值和标准差
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor

        self.files = []

    # 返回数据集中的样本数目
    def __len__(self):
        return len(self.files)

    # 图像和标签的预处理
    def input_transform(self, image, city=True):
        # 针对Cityscapes数据集，将图像从 BGR 转为 RGB
        if city:
            image = image.astype(np.float32)[:, :, ::-1]
        else:
            image = image.astype(np.float32)
        # 像素值进行归一化处理，并提供均值和标准差处理
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    # 标签转为 unit8 类型
    def label_transform(self, label):
        return np.array(label).astype(np.uint8)

    # 裁剪前，对图像进行填充
    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,pad_w, cv2.BORDER_CONSTANT, value=padvalue)
        return pad_image

    # 随机裁剪
    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    # 多尺度增强
    def multi_scale_aug(self, image, label=None, rand_scale=1, rand_crop=True):
        long_size = np.int32(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int32(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int32(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label= self.rand_crop(image, label)

        return image, label

    # 数据生成，返回生成的图像和标签
    def gen_sample(self, image, label, multi_scale=True, is_flip=True, city=True):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, rand_scale=rand_scale)

        image = self.input_transform(image, city=city)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image, label

    # 推理阶段，使用 model 对输入的 image 进行前向传播，得到预测结果 pred
    def inference(self, config, model, image):
        size = image.size()
        # 此处获取的 pred 不涉及损失函数的计算
        pred = model(image)
        pred = pred[0]
        pred = F.interpolate(input=pred, size=size[-2:], mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        # 将预测的对数概率转换回原始的概率值
        return pred.exp()