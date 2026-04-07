import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import os

## 损失函数的代码
# 适用于二分类的损失，直接传入logits（B，1，H，W）即可，BCE内部要做Sigmoid
class BceDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-5):
        """
        混合损失函数：BCE + Dice
        :param bce_weight: BCE 损失的权重
        :param dice_weight: Dice 损失的权重
        :param smooth: 防止分母为 0 的平滑项
        """
        super(BceDiceLoss, self).__init__()
        # 使用带 Logits 的 BCE，内部会自动做 Sigmoid，数值更稳定
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        :param logits: 模型的原始输出 (Batch, 1, H, W)，未经过 Sigmoid
        :param targets: 真实标签 (Batch, 1, H, W)，取值为 0 或 1
        """
        # 1. 计算 BCE 损失
        bce = self.bce_loss(logits, targets)

        # 2. 计算 Dice 损失
        # 先手动进行 Sigmoid，将输出映射到 [0, 1]
        probs = torch.sigmoid(logits)

        # 将数据展平为 (Batch, -1)
        batch_size = logits.size(0)
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        # 计算交集和并集
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        # 计算每一个 Batch 的 Dice 系数
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss = 1 - Dice 系数，然后取 Batch 的平均值
        dice = (1 - dice_score).mean()

        # 3. 总损失加权
        total_loss = self.bce_weight * bce + self.dice_weight * dice

        return total_loss


## 分割的评价指标：Dice、mIoU
# 标准做法是先整体计算一个batch里面的dice，然后跑完整个验证集后将所有的dice相加，然后除以验证集的总体数量，前景iou也是同理
def calculate_batch_dice(pred, target, threshold=0.5, smooth=1e-7):
    """
    计算 Batch 中每一张图片的 Dice 系数。
    参数:
        pred: 模型输出的 Logits, 尺寸 (B, 1, H, W)
        target: 真实标签, 尺寸 (B, 1, H, W), 值为 0 或 1
        threshold: 二值化阈值
        smooth: 平滑项，防止分母为 0
    返回:
        dices: torch.Tensor, 尺寸 (B,), 包含每张图的 Dice 结果
    """
    # 1. 激活与二值化
    probs = torch.sigmoid(pred)
    pred_mask = (probs > threshold).float()
    target = target.float()
    # 2. 展平为 (Batch_Size, 像素总数)
    batch_size = pred_mask.size(0)
    pred_flat = pred_mask.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    # 3. 计算交集和分母 (A + B)
    # intersection 尺寸: (B,)
    intersection = (pred_flat * target_flat).sum(dim=1)
    # cardinality 尺寸: (B,)
    cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    # 4. 计算每张图的 Dice (不取平均)
    dice = (2. * intersection + smooth) / (cardinality + smooth)
    return dice

def calculate_batch_iou(pred, target, threshold=0.5, smooth=1e-7):
    """
    计算 Batch 中每一张图片的前景 IoU。

    参数:
        pred: 模型输出的 Logits, 尺寸 (B, 1, H, W)
        target: 真实标签, 尺寸 (B, 1, H, W), 值为 0 或 1

    返回:
        ious: torch.Tensor, 尺寸 (B,), 包含每张图的前景 IoU 结果
    """
    # 1. 激活与二值化
    probs = torch.sigmoid(pred)
    pred_mask = (probs > threshold).float()
    target = target.float()
    # 2. 展平
    batch_size = pred_mask.size(0)
    pred_flat = pred_mask.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    # 3. 计算交集和并集
    # intersection 尺寸: (B,)
    intersection = (pred_flat * target_flat).sum(dim=1)
    # union 尺寸: (B,)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    # 4. 计算每张图的 IoU (不取平均)
    iou = (intersection + smooth) / (union + smooth)
    return iou

## 画图工具函数
def visualize_batch_results(image_batch, logits_batch, mask_batch, epoch, step, save_dir="visual_results", alpha=0.5):
    """
    随机从 Batch 中抽取 3 个样本，绘制 3x3 的可视化矩阵。

    参数:
    ----------
    image_batch:  torch.Tensor, 尺寸 (B, C, H, W), 原图 Batch
    logits_batch: torch.Tensor, 尺寸 (B, 1, H, W), 模型输出 Logits
    mask_batch:   torch.Tensor, 尺寸 (B, 1, H, W), 真实标签 Batch
    alpha:        float, 覆盖层的透明度
    """

    batch_size = image_batch.shape[0]
    # 随机选取 3 个不重复的索引
    indices = random.sample(range(batch_size), k=min(3, batch_size))

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # 内部处理函数：将 Tensor 转为显示的 Numpy
    def preprocess(t):
        t = t.detach().cpu()
        if t.shape[0] == 1:  # 单通道 (1, H, W) -> (H, W)
            return t.squeeze(0).numpy()
        else:  # 多通道 (C, H, W) -> (H, W, C)
            return t.permute(1, 2, 0).numpy()

    for i, idx in enumerate(indices):
        # 提取数据
        img = preprocess(image_batch[idx])
        # 归一化原图用于显示 (针对经过 Norm 处理的数据)
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        pred_prob = torch.sigmoid(logits_batch[idx])
        pred_mask = (pred_prob > 0.5).squeeze().cpu().numpy()
        gt_mask = mask_batch[idx].squeeze().cpu().numpy()

        # --- 第一列：原图 ---
        ax0 = axes[i, 0]
        ax0.imshow(img_norm, cmap='gray' if img_norm.ndim == 2 else None)
        ax0.set_title(f"Sample {idx} - Image")
        ax0.axis('off')

        # --- 第二列：预测覆盖 (绿色) ---
        ax1 = axes[i, 1]
        ax1.imshow(img_norm, cmap='gray' if img_norm.ndim == 2 else None)
        overlay_pred = np.zeros((*pred_mask.shape, 4))
        overlay_pred[pred_mask > 0] = [0, 1, 0, alpha]  # 绿色
        ax1.imshow(overlay_pred)
        ax1.set_title(f"Sample {idx} - Pred Overlay")
        ax1.axis('off')

        # --- 第三列：真实标签覆盖 (红色) ---
        ax2 = axes[i, 2]
        ax2.imshow(img_norm, cmap='gray' if img_norm.ndim == 2 else None)
        overlay_gt = np.zeros((*gt_mask.shape, 4))
        overlay_gt[gt_mask > 0] = [1, 0, 0, alpha]  # 红色
        ax2.imshow(overlay_gt)
        ax2.set_title(f"Sample {idx} - GT Overlay")
        ax2.axis('off')
    plt.tight_layout()

    # 2. 保存并关闭，释放内存
    save_path = os.path.join(save_dir, f"epoch_{epoch}_step_{step}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)  # 非常重要，防止显存/内存溢出
    print(f"Visual results saved to {save_path}")