import os
# --------------------------
# Configuration
# --------------------------
config = {
    "seed": 42,
    "run_eda": False,
    "data": {
        "base_dir": "./data",
        "train_img_subdir": "images/train2017",
        "val_img_subdir": "images/val2017",
        "label_subdir": "labels/annotations",
        "train_ann_file_original": "instances_train2017.json",
        "val_ann_file_original": "instances_val2017.json",
        "train_ann_file_person_only": "instances_train_person_only.json",
        "val_ann_file_person_only": "instances_val_person_only.json",
        "input_size": (640, 640)
    },
    "training": {
        "batch_size": 128,
        "epochs_to_run_this_session": 10,
        "initial_start_epoch_manual": 88,
        "lr": 0.0001,
        "weight_decay": 3e-4,
        "freeze_backbone": False
    },
    "experiment_name_template": "bs{bs}_lr{lr}",
    "weights_dir_base": "weights",
    "logs_dir_base": "experiments"
}
experiment_name = config["experiment_name_template"].format(bs=config["training"]["batch_size"], lr=config["training"]["lr"])

# Path
train_img_dir = os.path.join(config["data"]["base_dir"], config["data"]["train_img_subdir"])
val_img_dir = os.path.join(config["data"]["base_dir"], config["data"]["val_img_subdir"])
label_dir = os.path.join(config["data"]["base_dir"], config["data"]["label_subdir"])

train_ann_path_original = os.path.join(label_dir, config["data"]["train_ann_file_original"])
val_ann_path_original = os.path.join(label_dir, config["data"]["val_ann_file_original"])

train_ann_path_person_only = os.path.join(label_dir, config["data"]["train_ann_file_person_only"])
val_ann_path_person_only = os.path.join(label_dir, config["data"]["val_ann_file_person_only"])



import torch
# Env
from src.utils import set_seed
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Model, Criterion, Optimizer
from src.model import SimpleDetector
network = SimpleDetector()


# --------------------------
# Define Train Procedure
# --------------------------
from src.loader import get_custom_dataloaders
_, test_loader = get_custom_dataloaders(train_ann_path_person_only, train_img_dir, val_ann_path_person_only, val_img_dir, batch_size=10) 



network.to(device)
network.load_state_dict(torch.load("weights/bs128_lr0.0001/e_97.pt", map_location=device))
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
        
        draw_bboxes(image, pred, conf_threshold=0.7, save_path=f"outputs/pred_{i}.png")
        if i == 9:
            break  



