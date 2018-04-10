import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from enet_pytorch import enet_pytorch
from show import imshow

enet_pytorch.load_state_dict(torch.load('enet_pytorch.pth'))
enet_pytorch.cuda().eval()

img  = 'image.jpg'

im = cv2.imread(img).astype(np.float32)[:, :, ::-1]/255
e1 = cv2.getTickCount()

inp = torch.from_numpy(im.transpose(2,0,1)).unsqueeze(0).cuda()
out = enet_pytorch(Variable(inp))
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print(time)
print(out.data[0])
imshow(out.data[0])
