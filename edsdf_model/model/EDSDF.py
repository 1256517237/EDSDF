import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
# path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# sys.path.insert(0, path)
from .pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
# import decoder
from .decoder import EDSD
import sys
import os
# path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# sys.path.insert(0, path)


class EDSDF(nn.Module):# Efficient Deep Supervised Distillation Framework (EDSDF)
      def __init__(self, num_classes=1, kernel_sizes1=[1,3,5,7], kernel_sizes2=[1,3,5,7], expansion_factor=2, activation='relu6', encoder='pvt_v2_b0', pretrain=True):
            super(EDSDF, self).__init__()

            self.conv = nn.Sequential(
                  nn.Conv2d(1, 3, kernel_size=1),
                  nn.BatchNorm2d(3),
                  nn.ReLU(inplace=True)
            )

            # backbone network initialization with pretrained weight
            if encoder == 'pvt_v2_b0':
                  self.backbone = pvt_v2_b0()
                  path = '/pvt_v2_b0.pth'
                  channels=[256, 160, 64, 32]
            elif encoder == 'pvt_v2_b1':
                  self.backbone = pvt_v2_b1()
                  path = '/pvt_v2_b1.pth'
                  channels=[512, 320, 128, 64]
            elif encoder == 'pvt_v2_b2':
                  self.backbone = pvt_v2_b2()
                  path = '/pvt_v2_b2.pth'
                  channels=[512, 320, 128, 64]
            elif encoder == 'pvt_v2_b3':
                  self.backbone = pvt_v2_b3()
                  path = '/pvt_v2_b3.pth'
                  channels=[512, 320, 128, 64] 
            
            elif encoder == 'resnet18':
                  self.backbone = resnet18(pretrained=pretrain)
                  channels=[512, 256, 128, 64]
            elif encoder == 'resnet34':
                  self.backbone = resnet34(pretrained=pretrain)
                  channels=[512, 256, 128, 64]
            elif encoder == 'resnet50':
                  self.backbone = resnet50(pretrained=pretrain)
                  channels=[2048, 1024, 512, 256]
            elif encoder == 'resnet101':
                  self.backbone = resnet101(pretrained=pretrain)  
                  channels=[2048, 1024, 512, 256]
            elif encoder == 'resnet152':
                  self.backbone = resnet152(pretrained=pretrain)  
                  channels=[2048, 1024, 512, 256]
                  
            if pretrain==True and 'pvt_v2' in encoder:
                  save_model = torch.load(path)
                  model_dict = self.backbone.state_dict()
                  state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
                  model_dict.update(state_dict)
                  self.backbone.load_state_dict(model_dict)

            print('Model %s created, param count: %d' %(encoder+' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))

            self.decoder = EDSD(channels=channels, kernel_sizes1=kernel_sizes1, kernel_sizes2=kernel_sizes2, expansion_factor=expansion_factor, activation=activation)

            print('Model %s created, param count: %d' %('demodecoder: ', sum([m.numel() for m in self.decoder.parameters()])))
            
            self.mask_sh1 = nn.Conv2d(channels[0], num_classes, 1)
            self.mask_sh2 = nn.Conv2d(channels[1], num_classes, 1)
            self.mask_sh3 = nn.Conv2d(channels[2], num_classes, 1)
            self.mask_sh4 = nn.Conv2d(channels[3], num_classes, 1)

      def forward(self, x, mode='test'):

            if x.size()[1] == 1:
                  x = self.conv(x)
                  
            # encoder
            x1, x2, x3, x4 = self.backbone(x)

            # decoder
            mask_preds, boundary_preds = self.decoder(x4, [x3, x2, x1]) #

            # mask prediction heads
            mask_pred1 = self.mask_sh1(mask_preds[0])
            mask_pred2 = self.mask_sh2(mask_preds[1])
            mask_pred3 = self.mask_sh3(mask_preds[2])
            mask_pred4 = self.mask_sh4(mask_preds[3])

            mask_pred1 = F.interpolate(mask_pred1, scale_factor=32, mode='bilinear')
            mask_pred2 = F.interpolate(mask_pred2, scale_factor=16, mode='bilinear')
            mask_pred3 = F.interpolate(mask_pred3, scale_factor=8, mode='bilinear')
            mask_pred4 = F.interpolate(mask_pred4, scale_factor=4, mode='bilinear')

            # boundary pred
            boundary_pred1 = boundary_preds[0]
            boundary_pred2 = boundary_preds[1]
            boundary_pred3 = boundary_preds[2]
            boundary_pred4 = boundary_preds[3]

            boundary_pred1 = F.interpolate(boundary_pred1, scale_factor=32, mode='bilinear')
            boundary_pred2 = F.interpolate(boundary_pred2, scale_factor=16, mode='bilinear')
            boundary_pred3 = F.interpolate(boundary_pred3, scale_factor=8, mode='bilinear')
            boundary_pred4 = F.interpolate(boundary_pred4, scale_factor=4, mode='bilinear') 
            
            return [mask_pred1, mask_pred2, mask_pred3, mask_pred4], [boundary_pred1, boundary_pred2, boundary_pred3, boundary_pred4]                 



# if __name__ == '__main__':
#     model = EDSDF(expansion_factor = 1,use_se = False,activation= 'relu6',encoder = 'pvt_v2_b0').cuda()
#     input_tensor = torch.randn(1, 3, 224, 224).cuda()

#     outputs = model(input_tensor)
#     masks, boundaries = outputs

#     print(masks[0].size(), masks[1].size(), masks[2].size(), masks[3].size(),
#           boundaries[0].size(), boundaries[1].size(), boundaries[2].size(), boundaries[3].size())
                  
