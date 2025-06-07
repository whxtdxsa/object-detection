import os
import json

import torch

# --------------------------
# Configuration
# --------------------------
config = {
    'run_downloader': False,
    'run_eda': False,

    'batch_size': 256,
    'epochs': 10,
    'lr': 5e-4,
    'weight_decay': 3e-4,
    'freeze_backbone': False,
    'input_size': (640, 640),

    'resume': False,
    'start_epoch': 88,
}
experiment_name = f"bs{config['batch_size']}_lr{config['lr']}"
log_dir = f'./runs/train/{experiment_name}'

train_images_dir = './dataset/images/train'
train_annos_file = './dataset/labels/train.json'
val_images_dir = './dataset/images/val'
val_annos_file = './dataset/labels/val.json'

# --------------------------
# Env
# --------------------------
from src.utils import set_seed, get_amp_components 
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
amp_context, scaler = get_amp_components(device)

# --------------------------
# Data Preprocessing
# --------------------------
from src.downloader import prepare_dataset
if config['run_downloader']:
    prepare_dataset('train')
    prepare_dataset('val')


# --------------------------
# Experiment EDA
# --------------------------
from src.eda import (
    extract_annotations, plot_eda,
    get_image_id_to_image, get_image_id_to_annos, draw_bbox
)
if config['run_eda']:
    print('--- Running EDA ---')
    with open(val_annos_file, 'r') as f:
        eda_annos = json.load(f)

    extract_annotations(eda_annos)
    plot_eda(eda_annos)

    image_id_to_info = get_image_id_to_image(eda_annos) 
    image_id_to_annos = get_image_id_to_annos(eda_annos)

    id = list(image_id_to_info.keys())[0]
    image_file = os.path.join(val_images_dir, image_id_to_info[id]['file_name'])
    draw_bbox(image_file, image_id_to_annos[id], './asset/sample.png')
    print('--- EDA Finished ---')



# --------------------------
# Model, Criterion, Optimizer
# --------------------------
import torch.optim as optim

from src.model import ResNetFPN 
from src.loss import DetectionLoss
from src.utils import set_backbone_requires_grad
network = ResNetFPN().to(device)
criterion = DetectionLoss()
optimizer = optim.AdamW(network.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

if config['freeze_backbone']:
    set_backbone_requires_grad(network, requires_grad=False)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, network.parameters()), lr=config['lr'], weight_decay=config['weight_decay'])

start_epoch = 0
if config['resume']:
    start_epoch = config['start_epoch']
    weight_file = os.path.join(log_dir, f'e_{start_epoch}.pt')
    network.load_state_dict(torch.load(weight_to_file, map_location=device))

# --------------------------
# DataLoaders
# --------------------------
from src.loader import get_custom_dataloaders
train_loader, test_loader = get_custom_dataloaders(
    train_images_dir, train_annos_file, val_images_dir, val_annos_file, 
    batch_size=config['batch_size'], input_size=config['input_size']
) 

# --------------------------
# Training Loop 
# --------------------------
from src.trainer import train_one_epoch, evaluate_loss
from src.utils import init_csv_log, log_to_csv
csv_file = os.path.join(log_dir, 'result.csv') 
csv_fieldnames = ['epoch', 'train_loss', 'test_loss']
init_csv_log(csv_file, csv_fieldnames)

for i in range(config['epochs']):
    epoch = start_epoch + i + 1
    print(f"Epoch {epoch}/{start_epoch + config['epochs']}")
    train_loss = train_one_epoch(network, train_loader, optimizer, criterion, device, amp_context, scaler)
    test_loss = evaluate_loss(network, test_loader, criterion, device, amp_context)

    log_to_csv(csv_file, {
        'epoch': epoch,
        'train_loss': train_loss,
        'test_loss': test_loss
    })
    print(f'Train_loss: {train_loss:.4f}, Test_loss: {test_loss:.4f}')
    torch.save(network.state_dict(), os.path.join(log_dir, f'e_{epoch}.pt'))
