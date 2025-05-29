import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, size_average=True):
        #bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5 
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        loss = 1 - dice
        if size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target,size_average=True):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1#1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        loss = 0.5 * bce + (1 - dice)
        if size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): 
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): 
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  
            input = input.transpose(1,2)    
            input = input.contiguous().view(-1,input.size(2))   
        target = target.view(-1,1)

        logpt = F.log_softmax(input,dim=1)
        target = target.long()  
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.to(input.dtype)
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            loss = loss.mean()
            return loss
        else:
            loss = loss.sum()
            return loss

class CriterionKD(nn.Module):
    '''
    Knowledge Distillation Loss
    '''
    def __init__(self, temperature=1.0, reduction='mean'):
        super(CriterionKD, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, pred, soft):

        if self.reduction == 'none':
            p_s = torch.sigmoid(pred / self.temperature)
            p_t = torch.sigmoid(soft / self.temperature)
            loss = F.mse_loss(p_s, p_t, reduction=self.reduction)
        else:
            p_s = torch.sigmoid(pred / self.temperature).view(-1)
            p_t = torch.sigmoid(soft / self.temperature).view(-1)
            loss = F.mse_loss(p_s, p_t, reduction=self.reduction)

        return loss