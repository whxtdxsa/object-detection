import csv
import os

import torchvision.transforms.functional as F
import numpy as np
from contextlib import nullcontext
from PIL import ImageDraw
import random
import torch
import torchvision 
import pandas as pd
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_amp_components(device):
    if device.type == 'cuda':
        from torch.amp import autocast
        from torch.cuda.amp import GradScaler
        amp_context = autocast(device_type='cuda')
        scaler = GradScaler()
    else:
        amp_context = nullcontext()
        scaler = None
    return amp_context, scaler

def init_csv_log(fdst, fieldnames):
    file_exists = os.path.exists(fdst)
    if not file_exists:
        os.makedirs(os.path.dirname(fdst), exist_ok=True)
        with open(fdst, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

def log_to_csv(path, data_dict):
    with open(path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_dict.keys())
        writer.writerow(data_dict)

def plot_loss(path, fdst):
    df = pd.read_csv(path)

    plt.figure(figsize=(10, 6)) # Optional: Adjust figure size

    plt.plot(df['epoch'], df['train_loss'], label='train_loss')
    plt.plot(df['epoch'], df['test_loss'], label='test_loss')

    plt.title('Training and Test Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    
    plt.grid(True) # Optional: Add a grid
    plt.tight_layout()
    plt.savefig(fdst)
    plt.clf()


def draw_bboxes(image_tensor, pred_tensor, conf_threshold=0.1, save_path="output.jpg"):
    # pred_tensor[:,4] = torch.sigmoid(pred_tensor[:,4])
    img = F.to_pil_image(image_tensor.cpu())
    draw = ImageDraw.Draw(img)
    for box in pred_tensor:
        cx, cy, w, h, conf = box.tolist()
        if conf < conf_threshold:
            continue
        center_x_pixel = cx * img.width
        center_y_pixel = cy * img.height
        width_pixel = w * img.width
        height_pixel = h * img.height

        x1 = center_x_pixel - (width_pixel / 2)
        y1 = center_y_pixel - (height_pixel / 2)
        x2 = center_x_pixel + (width_pixel / 2)
        y2 = center_y_pixel + (height_pixel / 2)

        draw.rectangle([x1, y1, x2, y2], outline='red', width=6)
        draw.text((x1, y1 - 10), f"{conf:.2f}", fill="red")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    img.save(save_path)
    print(f"Saved: {save_path}")

def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(dim=1)
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

def postprocess_single_image_predictions(
    raw_preds_single_image: torch.Tensor,
    conf_threshold_nms_candidate: float = 0.2,
    iou_threshold_nms: float = 0.25
):
    """
    단일 이미지에 대한 모델의 원시 예측값을 후처리하여 NMS를 적용합니다.

    Args:
        raw_preds_single_image (torch.Tensor): [N, 5] 형태의 텐서.
            각 행은 [cx_norm, cy_norm, w_norm, h_norm, conf_logit].
        conf_threshold_nms_candidate (float): NMS를 적용하기 전에 사용할 1차 신뢰도 임계값.
                                               이 값 이상인 박스들만 NMS 대상으로 고려됩니다.
        iou_threshold_nms (float): NMS에 사용할 IoU 임계값.

    Returns:
        torch.Tensor: [M, 5] 형태의 텐서. NMS를 통과한 최종 검출 결과.
                      각 행은 [cx_norm, cy_norm, w_norm, h_norm, conf_probability].
                      NMS를 통과한 박스가 없으면 빈 텐서가 반환됩니다.
    """
    if raw_preds_single_image.numel() == 0:
        return torch.empty((0, 5), device=raw_preds_single_image.device, dtype=raw_preds_single_image.dtype)

    # 1. 좌표와 신뢰도 로짓 분리 및 신뢰도 확률 변환
    boxes_cxcywh_norm = raw_preds_single_image[:, :4]  # [N, 4]
    conf_logits = raw_preds_single_image[:, 4]         # [N]
    conf_probs = torch.sigmoid(conf_logits)            # [N]

    # 2. 1차 신뢰도 필터링 (NMS 후보 선정)
    candidate_indices = torch.where(conf_probs >= conf_threshold_nms_candidate)[0]

    if candidate_indices.numel() == 0: # 후보 박스가 없는 경우
        return torch.empty((0, 5), device=raw_preds_single_image.device, dtype=raw_preds_single_image.dtype)

    candidate_boxes_cxcywh = boxes_cxcywh_norm[candidate_indices] # [K, 4]
    candidate_scores = conf_probs[candidate_indices]              # [K]

    # 3. NMS를 위해 박스 형식을 xyxy로 변환
    candidate_boxes_xyxy = box_cxcywh_to_xyxy(candidate_boxes_cxcywh) # [K, 4]

    # 4. NMS 적용
    keep_indices = torchvision.ops.nms(candidate_boxes_xyxy, candidate_scores, iou_threshold_nms) # [M] (M <= K)

    # 5. NMS를 통과한 최종 박스와 점수 선택 (좌표는 원래의 cxcywh 형식 유지)
    final_boxes_cxcywh = candidate_boxes_cxcywh[keep_indices]
    final_scores = candidate_scores[keep_indices]

    if final_boxes_cxcywh.numel() == 0: # NMS 후 남은 박스가 없는 경우
        return torch.empty((0, 5), device=raw_preds_single_image.device, dtype=raw_preds_single_image.dtype)

    # 6. 최종 결과를 [M, 5] 형태로 재구성 ([cx,cy,w,h, conf_prob])
    final_detections = torch.cat([
        final_boxes_cxcywh,
        final_scores.unsqueeze(-1) # [M, 1] 형태로 변경
    ], dim=-1)

    return final_detections

