import json
import os

import albumentations as A
import cv2
import numpy as np
from PIL import Image
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

def get_custom_dataloaders(
    train_img_dir, train_ann_path, val_img_dir, val_ann_path,
    batch_size=128, input_size=(640, 640)
):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    IMG_WIDTH, IMG_HEIGHT = input_size
    letterbox_fill_color = (128, 128, 128)

    train_transform = A.Compose([
        # Letter Box
        A.LongestMaxSize(max_size=max(IMG_WIDTH, IMG_HEIGHT), p=1.0),
        A.PadIfNeeded(
            min_width=IMG_WIDTH,
            min_height=IMG_HEIGHT,
            border_mode=cv2.BORDER_CONSTANT,
            value=letterbox_fill_color,
            p=1.0
        ),

        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['category_ids'],
        min_visibility=0.1,
        min_area=100
    ))

    val_transform = A.Compose([
        # Letter Box
        A.LongestMaxSize(max_size=max(IMG_WIDTH, IMG_HEIGHT), p=1.0),
        A.PadIfNeeded(
            min_width=IMG_WIDTH,
            min_height=IMG_HEIGHT,
            border_mode=cv2.BORDER_CONSTANT,
            value=letterbox_fill_color,
            p=1.0
        ),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

    train_dataset = CustomDataset(train_ann_path, train_img_dir, transform=train_transform)
    test_dataset = CustomDataset(val_ann_path, val_img_dir, transform=val_transform)

    train_dataset = Subset(train_dataset, range(60000))
    test_dataset = Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True, collate_fn=custom_collate_fn)

    return train_loader, test_loader


class CustomDataset(Dataset):
    def __init__(self, annotation_path, image_dir, transform=None):
        with open(annotation_path, 'r') as f:
            self.coco = json.load(f)

        self.image_dir = image_dir
        self.transform = transform

        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.coco['images']}
        self.image_id_to_dims = {img['id']: (img['width'], img['height']) for img in self.coco['images']}

        self.annotations = self.coco['annotations']
        self.image_to_anns = {}
        for ann in self.annotations:
            self.image_to_anns.setdefault(ann['image_id'], []).append(ann)

        self.image_ids = list(self.image_to_anns.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, image_filename)
        
        img_pil = Image.open(image_path).convert('RGB')
        img_np = np.array(img_pil)
        
        original_width, original_height = self.image_id_to_dims[image_id]

        anns = self.image_to_anns.get(image_id, [])
        bboxes = []
        category_ids = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w<=0 or h <=0:
                print(f'warning: Skipping annotation')
                continue
            norm_x = (x + w / 2) / original_width
            norm_y = (y + h / 2) / original_height
            norm_w = w / original_width
            norm_h = h / original_height

            bboxes.append([norm_x, norm_y, norm_w, norm_h])
            category_ids.append(0)
            
        transformed = self.transform(image=img_np, bboxes=bboxes, category_ids=category_ids)
        img_transformed_tensor = transformed['image']
        transformed_bboxes = transformed['bboxes']
        if not transformed_bboxes:
            target = torch.empty((0, 4), dtype=torch.float32)
        else:
            target = torch.tensor(transformed_bboxes, dtype=torch.float32)


        return img_transformed_tensor, target
