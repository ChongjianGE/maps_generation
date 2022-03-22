import time
import os
import numpy as np
import torch
from torch.autograd import Variable
import cv2
import pytorch_ssim
from PIL import Image
import torchvision.transforms as transforms


gt_file ='./sample_images/target/'
results_file='./sample_images/generated_results/' 

transform = transforms.Compose([  \
      transforms.ToTensor(),   \
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])

loss_all = 0
step = 0
for name in os.listdir(gt_file):
    gt_img = transform(Image.open(os.path.join(gt_file,name))).unsqueeze(0)
    fake_img = transform(Image.open(os.path.join(results_file,name))).unsqueeze(0)


    gt_img = Variable(gt_img).cuda()
    fake_img = Variable(fake_img).cuda()

    loss_all = loss_all + pytorch_ssim.ssim(fake_img,gt_img)

    step += 1
    if step % 100 == 0:
        print('The current ssim score of step {} is {}'.format(step, loss_all/step))

print('-'*60)
print('The overall ssim score of step {} is {}'.format(step, loss_all/step))      
print('-'*60)
