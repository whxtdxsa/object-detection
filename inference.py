import os

import torch

# --------------------------
# Configuration
# --------------------------
config = {
    'batch_size': 16,
    'input_size': (640, 640),
    'conf_threshold': 0.75

}
weight_file = 'best_w.pt'
log_dir = './runs/predict/'

train_images_dir = './dataset/images/train'
train_annos_file = './dataset/labels/train.json'
val_images_dir = './dataset/images/val'
val_annos_file = './dataset/labels/val.json'

# --------------------------
# Env
# --------------------------
from src.utils import set_seed
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}\n')

# --------------------------
# Model
# --------------------------
from src.model import ResNetFPN 
network = ResNetFPN().to(device)
network.load_state_dict(torch.load(weight_file, map_location=device))

# --------------------------
# DataLoaders
# --------------------------
from src.loader import get_custom_dataloaders
train_loader, test_loader = get_custom_dataloaders(
    train_images_dir, train_annos_file, val_images_dir, val_annos_file, 
    batch_size=config['batch_size'], input_size=config['input_size']
) 

network.eval()
from src.utils import draw_bboxes, postprocess_single_image_predictions
with torch.no_grad():
    for i, (images, _) in enumerate(test_loader):
        images = images.to(device)
        preds = network(images)  # [1, 16, 5]
                
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
        
        image = (images[0].cpu() * std + mean).clamp(0, 1)
        pred = postprocess_single_image_predictions(preds[0])
        
        draw_bboxes(image, pred, conf_threshold=config['conf_threshold'], save_path=os.path.join(log_dir, f'pred_{i}.png'))
        if i == 9:
            break  



