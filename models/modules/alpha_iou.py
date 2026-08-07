"""Alpha-CIoU loss used by the MCA-YOLO-A ablation experiments."""

import math

import torch
from torch import nn

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss


def alpha_ciou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    alpha: float = 3.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return the Alpha-CIoU loss for aligned xyxy boxes.

    This follows the paper's extension of CIoU: each of the IoU, center-distance,
    and aspect-ratio penalty terms is raised to the same alpha power.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    w1 = (b1_x2 - b1_x1).clamp_min(eps)
    h1 = (b1_y2 - b1_y1).clamp_min(eps)
    w2 = (b2_x2 - b2_x1).clamp_min(eps)
    h2 = (b2_y2 - b2_y1).clamp_min(eps)

    inter = (
        (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_min(0)
        * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_min(0)
    )
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = (inter / union).clamp(0, 1)

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.square() + ch.square() + eps
    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2).square()
        + (b2_y1 + b2_y2 - b1_y1 - b1_y2).square()
    ) / 4

    v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).square()
    with torch.no_grad():
        ciou_alpha = v / (v - iou + 1 + eps)

    return (
        1 - iou.pow(alpha)
        + (rho2 / c2).clamp_min(0).pow(alpha)
        + (v * ciou_alpha).clamp_min(0).pow(alpha)
    )


class AlphaIoUBboxLoss(BboxLoss):
    """YOLOv8 box loss with the CIoU term replaced by Alpha-CIoU."""

    def __init__(self, reg_max: int = 16, alpha: float = 3.0):
        super().__init__(reg_max)
        if alpha <= 0:
            raise ValueError("Alpha-IoU 的 alpha 必须大于 0。")
        self.alpha = float(alpha)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        loss_iou = alpha_ciou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], alpha=self.alpha
        )
        loss_iou = (loss_iou * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = self._target_ltrb(anchor_points, target_bboxes)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = self._target_ltrb(anchor_points, target_bboxes) * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                torch.nn.functional.l1_loss(
                    pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none"
                ).mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

    def _target_ltrb(self, anchor_points: torch.Tensor, target_bboxes: torch.Tensor):
        """Keep target-distance conversion compatible across Ultralytics releases."""
        from ultralytics.utils.loss import bbox2dist

        return bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1 if self.dfl_loss else 16)


class AlphaIoUDetectionLoss(v8DetectionLoss):
    """YOLOv8 detection criterion using Alpha-CIoU for box regression."""

    def __init__(self, model: nn.Module, alpha: float = 3.0):
        super().__init__(model)
        self.alpha = float(alpha)
        self.bbox_loss = AlphaIoUBboxLoss(model.model[-1].reg_max, alpha).to(self.device)


def add_alpha_iou_loss(model: nn.Module, alpha: float = 3.0) -> None:
    """Install Alpha-IoU after the trainer has moved the model to its device."""

    def install(trainer):
        target_model = getattr(trainer.model, "module", trainer.model)
        target_model.criterion = AlphaIoUDetectionLoss(target_model, alpha)
        print(f"已启用 Alpha-CIoU 损失，alpha={alpha:g}")

    model.add_callback("on_pretrain_routine_end", install)

