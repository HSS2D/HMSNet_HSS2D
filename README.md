#### HSS2D Architecture：
![f7](https://github.com/user-attachments/assets/4c633e31-bdbe-41d2-b7f4-496db399895d)

#### Introduction:

Core code examples of HMSNet, including the **HSS2D** module based on **Mamba and HilbertScan**, the ZVM module combining Mamba and **Z-Order**, and the core code of the **VSS module**.

You can directly refer to and integrate our module code into your own models for training. 

Additionally, we provide complete loss function code, including **detail-optimized Dice loss** and **cross-entropy loss combined with Ohem online hard example mining**. 

We've also prepared **detailed dataset processing example code** to help you work efficiently.

For more information on Mamba, please visit: https://github.com/state-spaces/mamba.git

More on Vision Mamba: https://github.com/MzeroMiko/VMamba.git




#### Related Python Dependencies：

Below are the exact environment dependency versions for our module's full operation, provided for your reference. 
Install dependencies

We provide an exact requirements.txt for reproducibility:
```
pip install -r requirements.txt
```

Next, install the required dependencies for Mamba.
```
pip install causal_conv1d==1.1.1
pip install mamba-ssm==1.2.0.post1
git clone https://github.com/hustvl/Vim.git
# copy mamba-ssm dir in vim to conda env site-package dir
cp -rf mamba-1p1p1/mamba_ssm /opt/miniconda3/envs/mamba/lib/python3.10/site-packages
```

Verify whether the environment has been installed successfully. If no error is reported, the installation is successful.
```
import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=32,  # SSM state expansion factor
    d_conv=5,    # Local convolution width
    expand=3,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape
```


Next, test whether HSS2D is available.
```
import torch
import torch.nn as nn
from hvm import HSS2D

if __name__ == '__main__':
    hss2d = HSS2D(d_model=12).cuda()
    x = torch.randn(1, 12, 640, 640) # batch_size, channels, height, width
    x = x.cuda()
    y = ss2d(x)
    print(y.shape)
```



Pretrained Weights

We provide pretrained HVMBlock weights for quick evaluation:

- **Cityscapes**: [Download Link](https://pan.baidu.com/s/1Zt4bcJrIra_MvkOZqzPSZQ?pwd=apyk)  
- **CamVid**: [Download Link](https://pan.baidu.com/s/1-_fy_nlzMtE8XyNiFK99PA?pwd=63xx)  
- **ADE20K**: [Download Link](https://pan.baidu.com/s/1EV_WDmE-aCuHmuMmBhtoaw?pwd=c28h)  





