import torch
import lpips
from PIL import Image
import numpy as np


# ��������������������LPIPS�������������� image1 �� image2
# ������������
image1 = Image.open("image1.png")
image2 = Image.open("image2.png")

# ������������LPIPS����
lpips_model = lpips.LPIPS(net_type = "alex")

# ������������PyTorch��Tensor����
image1_tensor = torch.tensor(np.array(image1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
image2_tensor = torch.tensor(np.array(image2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

# ����LPIPS������������
distance = lpips_model(image1_tensor, image2_tensor)

print("LPIPS distance:", distance.item())

