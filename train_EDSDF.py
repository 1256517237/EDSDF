import sys
import os
path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, path)

import json
import time
import numpy as np
import torch
import torchvision
import random
from torchvision import datasets, transforms, models
from torch import nn
from torch.utils.data import DataLoader
from dataset_aug import preDataset_aug

from torchsummary import summary
import argparse
from loss.losses import DiceLoss
from thop import profile
from thop import clever_format
from medpy import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt

from edsdf_model.model.EDSDF import EDSDF

#torch.cuda.empty_cache()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#torch.cuda.empty_cache()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    

def seed_torch(seed: int = 1) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(False) 
    print(f"Random seed set as {seed}")

seed_torch()



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train(model, train_root,val_root,test_root,batch_size,lr,epochs, early_stop_patience=None):
    no_improve_count = 0
    
     # Lists to store metrics for visualization
    train_losses = []
    val_losses = []
    val_ious = []
    val_dices = []
    learning_rates = []

    # step1: data；
    with open(train_root, 'r') as fr:
         train_imgs = json.load(fr)
    with open(val_root, 'r') as fr:
         val_imgs = json.load(fr)
    with open(test_root, 'r') as fr:
         test_imgs = json.load(fr)

    train_data = preDataset_aug(train_imgs, shuffle=True, use_augmentation=True)
    val_data = preDataset_aug(val_imgs, shuffle=False, use_augmentation=False)
    test_data = preDataset_aug(test_imgs, shuffle=False, use_augmentation=False)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
        
    train_length=len(train_imgs)

    # step2: criterion and optimizer
    # set the loss function
    criterion_dice = DiceLoss()
    criterion_bce = nn.BCEWithLogitsLoss()


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)

    scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    best_dice = 0.0
    bestEpoch = 0

    # step3: training
    for epoch in range(epochs):

        trainLoss_sum = 0.0
        epochStart = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        model.train()
        for i, (img, mask, boundary) in enumerate(train_dataloader):
            # load data
            img = img.to(device)
            mask = mask.to(device).squeeze(0)
            boundary = boundary.to(device).squeeze(0)
            #print(img.shape, mask.shape)

            if mask.dim() == 3:  
                mask = mask.unsqueeze(1)
            if boundary.dim() == 3:  
                boundary = boundary.unsqueeze(1)

            outputs = model(img)
            pre_mask, pre_boundary = outputs # 
            #print(pre_mask[0].shape, pre_boundary[0].shape)
            
            #calculate loss
            loss_m1 = criterion_bce(pre_mask[0], mask) + criterion_dice(pre_mask[0], mask)
            loss_m2 = criterion_bce(pre_mask[1], mask) + criterion_dice(pre_mask[1], mask)
            loss_m3 = criterion_bce(pre_mask[2], mask) + criterion_dice(pre_mask[2], mask)
            loss_m4 = criterion_bce(pre_mask[3], mask) + criterion_dice(pre_mask[3], mask)

            loss_b1 = criterion_bce(pre_boundary[0], boundary)
            loss_b2 = criterion_bce(pre_boundary[1], boundary)
            loss_b3 = criterion_bce(pre_boundary[2], boundary)
            loss_b4 = criterion_bce(pre_boundary[3], boundary)

            loss = loss_m1 + loss_m2 + loss_m3 + loss_m4 + loss_b1 +loss_b2 + loss_b3 + loss_b4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainLoss_sum += loss.item()* img.size(0)
            
        scheduler.step()
        epochEnd = time.time()
        trainLossAvg = trainLoss_sum/train_length
        print("Training: Loss : {:.4f}".format(trainLossAvg))

        train_losses.append(trainLossAvg)

        learning_rates.append(optimizer.param_groups[0]['lr'])

        # valLossAvg = val_loss(model, val_dataloader, criterion_dice, criterion_bce)
        # print("Validation Loss : {:.4f}".format(valLossAvg))
        # val_losses.append(valLossAvg)
        val_losses.append(0)
        
        # step4: test;
        Dice, Iou= eval(model, val_dataloader)
        val_ious.append(Iou)
        val_dices.append(Dice)
        print("Epoch : {:03d},  IoU: {:.4f}, Dice: {:.4f}, Time: {:.4f}s".format(epoch, Iou, Dice, epochEnd-epochStart))

        dice_best = Dice
        if dice_best > best_dice :
            best_dice = dice_best
            bestEpoch = epoch
            no_improve_count = 0 
            torch.save(model.state_dict(), './EDSDF-S'+'.pth')
        else:
            no_improve_count += 1
            print(f"No improvement in Dice for {no_improve_count} epoch(s)")
        print("model for epoch {} saved".format(bestEpoch))
        print("*best Dice:{:.4f}".format(best_dice))    

        if no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in Dice for {early_stop_patience} epochs.")
            break
        

    # Visualization
    epochs = range(1, len(train_losses) + 1)
    visualize_metrics(train_losses, val_losses, val_ious, val_dices, learning_rates, epochs)
        
def val_loss(model, dataloader, criterion_dice, criterion_bce):
    # Set to evaluation mode
    model.eval()
    val_loss_sum = 0.0

    # Validation loop
    for j, (img, mask, boundary) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device)
        boundary = boundary.to(device)
        # Forward pass - compute outputs on input data using the model
        outputs = model(img)
        pre_mask, pre_boundary = outputs # 

        #calculate loss
        loss_m1 = criterion_bce(pre_mask[0], mask) + criterion_dice(pre_mask[0], mask)
        loss_m2 = criterion_bce(pre_mask[1], mask) + criterion_dice(pre_mask[1], mask)
        loss_m3 = criterion_bce(pre_mask[2], mask) + criterion_dice(pre_mask[2], mask)
        loss_m4 = criterion_bce(pre_mask[3], mask) + criterion_dice(pre_mask[3], mask)

        loss_b1 = criterion_bce(pre_boundary[0], boundary)
        loss_b2 = criterion_bce(pre_boundary[1], boundary)
        loss_b3 = criterion_bce(pre_boundary[2], boundary)
        loss_b4 = criterion_bce(pre_boundary[3], boundary)

        loss = loss_m1 + loss_m2 + loss_m3 + loss_m4 + loss_b1 +loss_b2 + loss_b3 + loss_b4
        val_loss_sum += loss.item()

    val_loss_avg = val_loss_sum / len(dataloader)
    return val_loss_avg

def visualize_metrics(train_losses, val_losses, val_ious, val_dices, learning_rates, epochs):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('losses.png')

    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_ious, label='Validation IoU', color='orange')
    plt.title('Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.savefig('val_ious.png')

    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_dices, label='Validation Dice', color='green')
    plt.title('Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.savefig('val_dices.png')

    plt.tight_layout()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, learning_rates, label='Learning Rate', color='red')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.savefig('learning_rate.png')

    plt.show()

def eval(model, dataloader):
    # Set to evaluation mode
    model.eval()

    dices = np.zeros(1)
    ious = np.zeros(1)

    eval_number = 0  

    # Validation loop
    for j, (img, mask, boundary) in enumerate(dataloader):
        img = img.to(device)
        mask = mask.to(device).squeeze(0)

        # Forward pass - compute outputs on input data using the model
        outputs = model(img)
        pre_masks, pre_boundaries = outputs
        pre_masks = pre_masks[3]

        pre_masks = torch.sigmoid(pre_masks)
        pre_masks = (pre_masks >= 0.5).type(torch.cuda.IntTensor)  

        bs = len(mask)
        for i in range(bs):
            pred_i = pre_masks[i].detach().cpu().numpy()
            true_i = mask[i].detach().cpu().numpy()

            dice_i = metric.binary.dc(pred_i, true_i)
            iou_i = metric.binary.jc(pred_i, true_i)

            dices[0] += dice_i
            ious[0] += iou_i

        eval_number += bs  

    dices = dices / eval_number
    ious = ious / eval_number

    return dices[0], ious[0]

def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(lr)
    return lr


if __name__=='__main__':
    print(device)
    ap = argparse.ArgumentParser()
    ap.add_argument("-trd", "--train_root", default = './train.json',
		help = "test image path")
    ap.add_argument("-val", "--val_root", default='./val.json',
                    help="test image path")
    ap.add_argument("-ted", "--test_root", default = './test.json',
		help = "test image path")
    ap.add_argument("-b", "--batch_size", type=int, default=32,
		help="training batch size")
    ap.add_argument('-l', '--learning_rate', type=float, default=1e-3,
    	help='Learning rate')#1e-3
    ap.add_argument("-e", "--epochs", type=int, default=200,
		help="no. of epochs")

    args = vars(ap.parse_args())

    train_root = args["train_root"]
    val_root = args["val_root"]
    test_root = args["test_root"]

    model = EDSDF(expansion_factor = 1,encoder = 'pvt_v2_b0')

    #print(model)
    # model = model.to(device)

    # Print the model to be trained
    # summary(model, input_size=(3, 224, 224), batch_size=args["batch_size"])
    # linput = torch.randn(1, 3, 224, 224).to(device)
    # flops, params = profile(model, inputs=(input,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(fops, params)

    # encoder = model.backbone  
    # decoder = model.decoder  

    # encoder_flops, encoder_params = profile(encoder, inputs=(input,))
    # encoder_flops, encoder_params = clever_format([encoder_flops, encoder_params], "%.6f")
    # print(f"Encoder - FLOPs: {encoder_flops}, Params: {encoder_params}")
    # x1, x2, x3, x4 = encoder(input)
    # skips = [x3, x2, x1]

    # decoder_flops, decoder_params = profile(decoder, inputs=(x4,skips))
    # decoder_flops, decoder_params = clever_format([decoder_flops, decoder_params], "%.6f")
    # print(f"Decoder - FLOPs: {decoder_flops}, Params: {decoder_params}")

    # Train the model
    train(model, train_root=train_root, val_root=val_root, test_root=test_root,
          batch_size=args["batch_size"],lr=args["learning_rate"],epochs=args["epochs"], early_stop_patience=50)
