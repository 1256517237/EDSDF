#coding:utf-8
import os
import json
import time
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn
from torch.utils.data import DataLoader
from dataset_aug_demodecoder import preDataset_aug
from torchsummary import summary
import argparse
from thop import profile
from thop import clever_format
from medpy import metric

from edsdf_model.model.EDSDF import EDSDF


torch.cuda.empty_cache()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, test_root, batch_size):
    # step1: data
    with open(test_root, 'r') as fr:
        test_imgs = json.load(fr)
    test_data = preDataset_aug(test_imgs, shuffle=False, use_augmentation=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()

    dices = np.zeros(1)
    ious = np.zeros(1)
    accs = np.zeros(1)
    ses = np.zeros(1)
    sps = np.zeros(1)
    hds = np.zeros(1)

    eval_number = 0  

    # Validation loop
    for j, (img, mask, boundary) in enumerate(test_dataloader):
        img = img.to(device)
        mask = mask.to(device)

        bs = len(mask)

        # Forward pass - compute outputs on input data using the model
        outputs = model(img)
        pre_masks, pre_boundaries = outputs
        pre_masks = pre_masks[3]
        pre_masks = torch.sigmoid(pre_masks)
        pre_masks = (pre_masks >= 0.5).type(torch.cuda.IntTensor)  

        for i in range(bs):
            pre_mask = pre_masks[i].squeeze(0).detach().cpu().numpy()
            true_mask = mask[i].squeeze(0).cpu().numpy()

            dice_i = metric.binary.dc(pre_mask, true_mask)
            iou = metric.binary.jc(pre_mask, true_mask)
            acc = Accuracy(pre_mask, true_mask)
            se = metric.binary.sensitivity(pre_mask, true_mask)
            sp = metric.binary.specificity(pre_mask, true_mask)

            if true_mask.sum() == 0 or pre_mask.sum() == 0:
                hd95 = 0
            else:
                hd95 = metric.binary.hd95(pre_mask, true_mask)

            dices[0] += dice_i
            ious[0] += iou
            accs[0] += acc
            ses[0] += se
            sps[0] += sp
            hds[0] += hd95

        eval_number += bs  


    dices = dices / eval_number
    ious = ious / eval_number
    accs = accs / eval_number
    ses = ses / eval_number
    sps = sps / eval_number
    hds = hds / eval_number

    return accs[0], ses[0], sps[0], dices[0], ious[0], hds[0]

def Accuracy(output, input):

    input = input.astype('int')
    output = output.reshape(-1)
    input = input.reshape(-1)

    return (output == input).sum() / len(output)

def ignore_extra_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith('total_ops') and not key.startswith('total_params'):
            new_state_dict[key] = value
    return new_state_dict

if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-ted", "--test_root", default = './test.json',
		help = "test image path")
    ap.add_argument("-b", "--batch_size", type=int, default=1,
		help="training batch size")
    args = vars(ap.parse_args())

    test_root = args["test_root"]

    directory = "saved_pth/"
    
    model = EDSDF(expansion_factor = 1,encoder = 'pvt_v2_b2')

    model = model.to(device)
    input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

    model.load_state_dict(torch.load(directory + 'saved_pth' + '.pth'), strict=False)

    Acc, Sen, Spe, Dice, IoU, HD= test(model, test_root=test_root, batch_size=args["batch_size"])

    print(Acc)
    print(Sen)
    print(Spe)
    print(Dice)
    print(IoU)
    print(HD)
