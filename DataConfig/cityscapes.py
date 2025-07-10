import os
import cv2
import numpy as np
from PIL import Image
import torch
from .base_dataset import BaseDataset

class Cityscapes(BaseDataset):
    def __init__(self,
                 # root: 数据集根目录，存储图像和标签。
                 root,
                 # list_path: 图像文件列表路径。
                 list_path,
                 num_classes=19,
                 # 使用多尺度训练
                 multi_scale=True,
                 # 使用翻转
                 flip=True,
                 # 忽略标签的像素值
                 ignore_label=255,
                 # 基准图像大小
                 base_size=2048,
                 # 裁剪的图像尺寸
                 crop_size=(512, 1024),
                 # 随机尺度因子
                 scale_factor=16,
                 # 图像标准化的均值和方差
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 other_scale=False
                 ):

        # Cityscapes 类继承自 BaseDataset
        # 通过 super() 调用父类的构造函数来初始化一些公共的参数（如 ignore_label、base_size、crop_size 等）
        # 这些参数在 BaseDataset 类中定义并处理。
        super(Cityscapes, self).__init__(ignore_label, base_size, crop_size, scale_factor, mean, std, other_scale)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()

        # 用于标签类别转换的字典
        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}

        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                                        1.0166, 0.9969, 0.9754, 1.0489,
                                        0.8786, 1.0023, 0.9539, 0.9843, 
                                        1.1116, 0.9037, 1.0865, 1.0955, 
                                        1.0865, 1.1529, 1.0507]).cuda()

    # 读取文件列表
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files

    # 为了适合训练，将原始标签中的类别值转换为新的标签值
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    # __getitem__会自动调用，即在使用数据加载器（DataLoader）时系统自动调用
    # DataLoader 会迭代数据集对象，逐个获取数据样本
    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,'cityscapes',item["img"]), cv2.IMREAD_COLOR)
        size = image.shape

        # 如果是测试集，仅对图像进行预处理，并返回图像及其原始尺寸
        # 这是因为 Cityscapes 官方没有提供测试集标签
        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            #print(f"--------> __getitem__ image 尺寸 {image.shape}")
            # 此处在 func_train_eval_test.py 中的 test 中调用
            return image.copy(), np.array(size), name

        # 如果是训练集或验证集
        # 对图像和标签进行转换，调用 gen_sample 方法进行数据增强
        label = cv2.imread(os.path.join(self.root,'cityscapes',item["label"]), cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)
        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)
        #print(f"--------> __getitem__ image 尺寸 {image.shape}")
        # 返回：原始图像，标签，尺寸，名称
        return image.copy(), label.copy(), np.array(size), name

    # 调用父类中的 inference 方法来得到预测结果
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    # 将预测结果转换为每个像素类别的标签，然后保存为 PNG 图像。
    # 使用 np.argmax(preds, axis=1) 选择每个像素的最大类别索引作为预测类别。
    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))