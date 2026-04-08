import os
import sys
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 路径修复
_JOINT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _JOINT_ROOT not in sys.path:
    sys.path.insert(0, _JOINT_ROOT)

from datasets.ISIC2018_dataset import ISIC2018Dataset
from models.IrModel.ConvIR import build_net
from models.SegModel.VMUnet import VMUNet
from utils import calculate_batch_dice, calculate_batch_iou, BceDiceLoss


# ================= 基础工具 =================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('JointLight')
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ================= Loss =================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        d = x - y
        return torch.mean(torch.sqrt(d * d + self.eps * self.eps))


class EdgeAwareAlignmentLoss(nn.Module):
    """边缘感知对齐损失 - 更合理的梯度促进方式"""

    def __init__(self, edge_weight=2.0):
        super().__init__()
        self.edge_weight = edge_weight

    def forward(self, x_hat, img_hr, mask, logits):
        """
        x_hat: 去噪后的图像
        img_hr: 高清目标图像
        mask: 分割真值
        logits: 分割预测logits
        """
        # 计算重建误差
        rec_error = (x_hat - img_hr).abs()

        # 计算分割概率
        seg_prob = torch.sigmoid(logits)

        # 边缘区域权重（分割边界附近）
        edge_mask = self._get_edge_mask(mask)

        # 在分割区域赋予更高权重
        region_weight = 1.0 + self.edge_weight * edge_mask

        # 对齐损失：重建误差 × 分割注意力
        align_loss = torch.mean(rec_error * region_weight * (1.0 + seg_prob))

        return align_loss

    def _get_edge_mask(self, mask, kernel_size=3):
        """提取边缘区域"""
        # 使用 Sobel 算子或简单差分
        with torch.no_grad():
            edge = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :])
            edge = torch.nn.functional.pad(edge, (0, 0, 0, 1))
            return (edge > 0.5).float()


# ================= 主函数 =================
def main():
    # ====== 配置 ======
    data_path = os.path.join(_JOINT_ROOT, 'datasets', 'ISIC2018')
    out_dir = os.path.join(_JOINT_ROOT, 'results', 'train_joint_light')

    pre_train_pth = ''
    ir_pretrained_pth = ''

    batch_size = 1
    num_epochs = 100
    accumulation_steps = 2  # 梯度累积，进一步节省显存

    lr_ir = 5e-5
    lr_seg = 1e-4

    lambda_align = 0.05  # 对齐损失权重（调低一点更稳定）
    use_edge_aware = True  # 使用边缘感知对齐

    sigma = 0.2
    seed = 42
    patience = 15

    set_seed(seed)
    logger = get_logger(out_dir)

    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    if torch.cuda.is_available():
        logger.info(f'初始显存: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB')

    # ====== 数据 ======
    train_loader = DataLoader(
        ISIC2018Dataset(data_path, split='train', image_size=160, sigma=sigma),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    valid_loader = DataLoader(
        ISIC2018Dataset(data_path, split='test', image_size=160, sigma=sigma),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # ====== 模型 ======
    ir_model = build_net(version='base', in_nc=3, out_nc=3).to(device)

    if ir_pretrained_pth and os.path.isfile(ir_pretrained_pth):
        ir_model.load_state_dict(torch.load(ir_pretrained_pth, map_location=device))
        logger.info(f'加载去噪预训练: {ir_pretrained_pth}')

    seg_model = VMUNet(num_classes=1, load_ckpt_path=pre_train_pth if pre_train_pth else None).to(device)

    if pre_train_pth:
        seg_model.load_from()

    # ❗冻结分割模型（关键省显存）
    seg_model.eval()
    for p in seg_model.parameters():
        p.requires_grad = False
    logger.info('分割模型已冻结，不参与训练')

    # ====== Loss ======
    loss_rec_fn = CharbonnierLoss().to(device)

    if use_edge_aware:
        align_loss_fn = EdgeAwareAlignmentLoss(edge_weight=2.0).to(device)
    else:
        align_loss_fn = None

    # 注意：分割损失只用于监控，不参与训练
    loss_seg_fn = BceDiceLoss().to(device)

    optimizer = optim.AdamW(
        ir_model.parameters(),
        lr=lr_ir,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    best_dice = 0.0
    best_path = os.path.join(ckpt_dir, 'best_joint_light.pth')
    no_improve = 0

    logger.info(f'轻量联合训练 | lambda_align={lambda_align} | edge_aware={use_edge_aware}')

    # ================= 训练 =================
    for epoch in range(1, num_epochs + 1):
        ir_model.train()

        total_loss = 0.0
        total_rec_loss = 0.0
        total_align_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)

        optimizer.zero_grad()

        for step, (img_lq, img_hr, mask, _) in enumerate(pbar):
            img_lq = img_lq.to(device, non_blocking=True)
            img_hr = img_hr.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # ====== forward ======
            x_hat = ir_model(img_lq)[-1]  # [B, 3, H, W]
            l_rec = loss_rec_fn(x_hat, img_hr)

            # segmentation（只用于计算对齐损失，不反传）
            with torch.no_grad():
                logits = seg_model(x_hat)
                l_seg_monitor = loss_seg_fn(logits, mask)  # 仅用于监控
                seg_prob = torch.sigmoid(logits)

            # ====== ⭐ 对齐损失（修复版） ======
            if use_edge_aware and lambda_align > 0:
                # 边缘感知对齐损失
                align_loss = align_loss_fn(x_hat, img_hr, mask, logits)
            else:
                # 简单对齐损失（修复维度问题）
                # 在分割区域（概率>0.5）给重建误差更高权重
                region_weight = 1.0 + seg_prob  # [B, 1, H, W]
                align_loss = torch.mean((x_hat - img_hr).abs() * region_weight)

            # ====== 总损失 ======
            loss = l_rec + lambda_align * align_loss

            # 梯度累积
            loss = loss / accumulation_steps
            loss.backward()

            # 每 accumulation_steps 步更新一次
            if (step + 1) % accumulation_steps == 0:
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(ir_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            total_rec_loss += l_rec.item()
            total_align_loss += align_loss.item()

            pbar.set_postfix(
                rec=f"{l_rec.item():.4f}",
                seg_monitor=f"{l_seg_monitor.item():.4f}",
                align=f"{align_loss.item():.4f}"
            )

            # 定期清理缓存
            if step % 50 == 0:
                torch.cuda.empty_cache()

        # 处理最后可能未更新的梯度
        if (step + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(ir_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        avg_rec_loss = total_rec_loss / len(train_loader)
        avg_align_loss = total_align_loss / len(train_loader)

        scheduler.step()

        # ================= 验证 =================
        ir_model.eval()
        seg_model.eval()

        val_dice = 0.0
        val_iou = 0.0
        val_psnr = 0.0

        with torch.no_grad():
            for img_lq, img_hr, mask, _ in valid_loader:
                img_lq = img_lq.to(device, non_blocking=True)
                img_hr = img_hr.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                x_hat = ir_model(img_lq)[-1]
                logits = seg_model(x_hat)

                val_dice += calculate_batch_dice(logits, mask).mean().item()
                val_iou += calculate_batch_iou(logits, mask).mean().item()

                # 计算 PSNR（监控重建质量）
                mse = torch.mean((x_hat - img_hr) ** 2)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
                val_psnr += psnr.item()

        val_dice /= len(valid_loader)
        val_iou /= len(valid_loader)
        val_psnr /= len(valid_loader)

        logger.info(
            f'Epoch {epoch:03d} | Loss {avg_loss:.4f} (Rec:{avg_rec_loss:.4f} Align:{avg_align_loss:.4f}) | '
            f'Dice {val_dice:.4f} IoU {val_iou:.4f} PSNR {val_psnr:.2f}dB'
        )

        # 打印显存使用
        if torch.cuda.is_available() and epoch % 10 == 0:
            logger.info(f'显存使用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB')

        # ====== 保存 ======
        if val_dice > best_dice:
            best_dice = val_dice
            no_improve = 0

            torch.save({
                'ir': ir_model.state_dict(),
                'epoch': epoch,
                'dice': best_dice,
                'psnr': val_psnr
            }, best_path)

            logger.info(f'✔ 保存最佳模型 Dice={best_dice:.4f}')
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f'早停触发，最佳 Dice={best_dice:.4f}')
                break

        # 清理缓存
        torch.cuda.empty_cache()

    logger.info(f'训练结束，最佳 Dice={best_dice:.4f}')


if __name__ == '__main__':
    main()