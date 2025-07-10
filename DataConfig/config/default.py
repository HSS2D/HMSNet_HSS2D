# 通过 __future__ 导入，确保兼容 Python 2 和 Python 3 的不同语法行为
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# 使用 YACS 库来组织和管理深度学习训练任务的配置。
# YACS（Yet Another Configuration System）是一个用于简化和组织配置文件的库，它支持以 YAML 格式定义配置，
from yacs.config import CfgNode as CN
# 使用 YACS 库中的 CfgNode 来创建一个空的配置对象 _C，此对象将用于管理所有配置项。
_C = CN()
# 分别用于存储输出文件和日志文件的路径，当前为空
_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
# 指定使用的 GPU 的索引（默认为第一个 GPU）
_C.GPUS = (0,)
# 数据加载时使用的工作线程数量（默认为 4）
_C.WORKERS = 4
# 训练过程中日志打印的频率（每 20 次打印一次）
_C.PRINT_FREQ = 20
# 是否在中断后自动恢复训练，默认为 False
_C.AUTO_RESUME = False
# 是否将数据加载到固定内存中
_C.PIN_MEMORY = True

# 控制是否启用 CUDNN 库，True 表示启用
_C.CUDNN = CN()
_C.CUDNN.ENABLED = True
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False

# 模型的配置
_C.MODEL = CN()
_C.MODEL.NAME = ''
_C.MODEL.PRETRAINED = ''
# 如果设置为 True，会保持图像的对齐，这个配置通常影响上采样操作。
_C.MODEL.ALIGN_CORNERS = True
# 模型输出的层数，默认为 1
_C.MODEL.NUM_OUTPUTS = 1

# 损失函数的配置
_C.LOSS = CN()
# 是否使用 OHEM（Online Hard Example Mining），旨在通过优先训练困难样本来提高模型性能
_C.LOSS.USE_OHEM = True
# OHEM 的阈值，表示仅选择预测置信度低于该阈值的样本进行训练
_C.LOSS.OHEMTHRES = 0.7
# 每个批次中保留的样本数
_C.LOSS.OHEMKEEP = 100000
# 是否使用类平衡，默认为False
_C.LOSS.CLASS_BALANCE = False
# 类平衡的权重设置（通常用于处理类别不均衡的情况)
_C.LOSS.BALANCE_WEIGHTS = [0.5, 0.5]
# 边界损失相关的权重
_C.LOSS.SB_WEIGHTS = 0.5

# 数据集
_C.DATASET = CN()
_C.DATASET.ROOT = 'data/'
_C.DATASET.DATASET = 'cityscapes'
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.TRAIN_SET = 'list/cityscapes/train.lst'
_C.DATASET.EXTRA_TRAIN_SET = ''
_C.DATASET.TEST_SET = 'list/cityscapes/val.lst'

# 训练相关配置
_C.TRAIN = CN()
# 训练图像大小
_C.TRAIN.IMAGE_SIZE = [1024, 1024]  # width * height
# 输入图像基础大小
_C.TRAIN.BASE_SIZE = 2048
# 随机水平翻转增强
_C.TRAIN.FLIP = True
# 多尺度训练
_C.TRAIN.MULTI_SCALE = True
# 多尺度缩放因子
_C.TRAIN.SCALE_FACTOR = 16
# 学习率
_C.TRAIN.LR = 0.01
# 额外学习率
_C.TRAIN.EXTRA_LR = 0.001
# 优化器类型
_C.TRAIN.OPTIMIZER = 'sgd'
# 动量
_C.TRAIN.MOMENTUM = 0.9
# 权重衰减
_C.TRAIN.WD = 0.0001
# 是否启动 Nesterov 动量
_C.TRAIN.NESTEROV = False
# 忽视的标签
_C.TRAIN.IGNORE_LABEL = -1
# epoch的设置
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.EXTRA_EPOCH = 0
_C.TRAIN.RESUME = False
# 每个 GPU 上的批次
_C.TRAIN.BATCH_SIZE_PER_GPU = 8
# 是否打乱数据
_C.TRAIN.SHUFFLE = True

# 测试相关设置
_C.TEST = CN()
# 测试图像大小
_C.TEST.IMAGE_SIZE = [2048, 1024]  # width * height
_C.TEST.BASE_SIZE = 2048
_C.TEST.BATCH_SIZE_PER_GPU = 1
_C.TEST.MODEL_FILE = ''
_C.TEST.FLIP_TEST = False
_C.TEST.MULTI_SCALE = False

_C.TEST.OUTPUT_INDEX = -1


# 更新配置的函数，传入新的配置文件，然后进行配置更新
def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)