import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, complete_box_iou_loss
from src.utils import box_cxcywh_to_xyxy

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none') 

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        p_t = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - p_t)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DetectionLoss(nn.Module):
    def __init__(self, lambda_box=7.5, lambda_conf=3.0, iou_thresh=0.30, use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_conf = lambda_conf
        self.iou_thresh = iou_thresh
        
        if use_focal_loss:
            self.conf_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.conf_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        total_loss = torch.tensor(0.0, device=preds[0].device)
        num_preds_in_batch = len(preds)

        accumulated_conf_loss = torch.tensor(0.0, device=preds[0].device)
        accumulated_box_loss = torch.tensor(0.0, device=preds[0].device)
        num_images_with_positives_for_box_loss = 0

        for pred, target in zip(preds, targets):
            pred_conf_logits = pred[:, 4]  # [N]
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred[:, :4])  # [N, 4]

            target = target.to(pred.device)
            conf_target = torch.zeros_like(pred_conf_logits)
            target_boxes_xyxy = box_cxcywh_to_xyxy(target[:, :4])  # [M, 4]

            matched_mask = torch.zeros_like(pred_conf_logits, dtype=torch.bool)

            if target_boxes_xyxy.size(0) > 0:
                ious = box_iou(pred_boxes_xyxy, target_boxes_xyxy)  # [N, M]
                best_iou, best_gt = ious.max(dim=1)
                matched_mask = best_iou > self.iou_thresh
                conf_target[matched_mask] = 1 

            conf_loss = self.conf_loss_fn(pred_conf_logits, conf_target)
            accumulated_conf_loss += conf_loss

            if matched_mask.any():
                pos_pred_boxes = pred_boxes_xyxy[matched_mask]
                pos_target_boxes = target_boxes_xyxy[best_gt[matched_mask]]
                box_loss = complete_box_iou_loss(pos_pred_boxes, pos_target_boxes, reduction='mean')
                accumulated_box_loss += box_loss
                num_images_with_positives_for_box_loss += 1
        
            if torch.rand(1).item() < 0.0001:
                print(f"[DEBUG] conf_target > 0: {(conf_target > 0).sum().item()} / {len(conf_target)}")
                print(f"conf_target mean: {conf_target.mean().item():.4f}")
                print(f"[DEBUG] pred_conf range: min={pred_conf_logits.min().item():.3f}, max={pred_conf_logits.max().item():.3f}")

        final_conf_loss = accumulated_conf_loss / num_preds_in_batch
        if num_images_with_positives_for_box_loss > 0:
            final_box_loss = accumulated_box_loss / num_images_with_positives_for_box_loss
        else:
            final_box_loss = torch.tensor(0.0, device=preds[0].device)
        total_loss += self.lambda_conf * final_conf_loss + self.lambda_box * final_box_loss
        if torch.rand(1).item() < 0.005:
            print(f"[DEBUG] conf_loss: {final_conf_loss.item()}")
            print(f"[DEBUG] box_loss: {final_box_loss.item()}")
            print(f"[DEBUG] total_loss: {total_loss.item()}")

        return total_loss
