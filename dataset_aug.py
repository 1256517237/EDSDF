import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import rotate, InterpolationMode

class preDataset_aug(Dataset):
    def __init__(self, root, shuffle=True, use_augmentation=True):
        if shuffle:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.use_augmentation = use_augmentation  

        # Compute mean and std
        mean, std = self.compute_mean_std(root)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.mask_transform = transforms.Compose([transforms.ToTensor()])
        self.boundary_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < len(self), 'index range error'

        img_path = self.lines[index]
        img = Image.open(img_path).convert('RGB')
        mask_path = img_path.replace('images', 'masks')
        mask = Image.open(mask_path).convert('L')
        boundary_path = img_path.replace('images', 'boundaries')
        boundary = Image.open(boundary_path).convert('L')

        if self.use_augmentation:  
            img, mask, boundary = self.apply_transforms(img, mask, boundary)

        img = self.transform(img)
        mask = self.mask_transform(mask)
        boundary = self.boundary_transform(boundary)
        #print(f"Mask shape: {mask.shape}")

        return img, mask, boundary


    def apply_transforms(self, img, mask, boundary):
        if random.random() > 0.5:
            img, mask, boundary = self.random_rot_flip(img, mask, boundary)
        if random.random() > 0.5:
            img, mask, boundary = self.random_rotate(img, mask, boundary)
        
        return img, mask, boundary

    def random_rot_flip(self, img, mask, boundary):
        k = random.randint(0, 3)
        img = TF.rotate(img, 90 * k)
        mask = TF.rotate(mask, 90 * k)
        boundary = TF.rotate(boundary, 90 * k)
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            boundary = TF.hflip(boundary)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
            boundary = TF.vflip(boundary)
        return img, mask, boundary

    def random_rotate(self, img, mask, boundary):
        angle = random.randint(-30, 30)
        img = TF.rotate(img, angle)
        mask = TF.rotate(mask, angle)
        boundary = TF.rotate(boundary, angle)
        return img, mask, boundary

    def compute_mean_std(self, root):
        all_images = [Image.open(img_path).convert('RGB') for img_path in root]
        all_images = [transforms.ToTensor()(img) for img in all_images]
        stacked_images = torch.stack(all_images)
        mean = torch.mean(stacked_images, dim=(0, 2, 3))
        std = torch.std(stacked_images, dim=(0, 2, 3))
        return mean.tolist(), std.tolist()
