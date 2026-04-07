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
    logger = logging.getLogger('JointSimple')
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


# ================= 主函数 =================
def main():
    # ====== 配置 ======
    data_path = os.path.join(_JOINT_ROOT, 'datasets', 'ISIC2018')
    out_dir = os.path.join(_JOINT_ROOT, 'results', 'train_joint_align')

    pre_train_pth = ''         # VMUNet 预训练（可选）
    ir_pretrained_pth = ''     # 去噪预训练（强烈建议填）

    batch_size = 8
    num_epochs = 100

    lr_ir = 5e-5
    lr_seg = 1e-4

    lambda_seg = 0.5
    lambda_align = 0.1   # ⭐ 梯度促进强度（创新点）

    sigma = 0.2
    seed = 42
    patience = 15

    set_seed(seed)
    logger = get_logger(out_dir)

    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ====== 数据 ======
    train_loader = DataLoader(
        ISIC2018Dataset(data_path, split='train', image_size=256, sigma=sigma),
        batch_size=batch_size, shuffle=True, num_workers=0
    )

    valid_loader = DataLoader(
        ISIC2018Dataset(data_path, split='val', image_size=256, sigma=sigma),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    # ====== 模型 ======
    ir_model = build_net(version='base', in_nc=3, out_nc=3).to(device)
    if ir_pretrained_pth and os.path.isfile(ir_pretrained_pth):
        ir_model.load_state_dict(torch.load(ir_pretrained_pth, map_location=device))
        logger.info(f'加载去噪预训练: {ir_pretrained_pth}')

    seg_model = VMUNet(num_classes=1, load_ckpt_path=pre_train_pth if pre_train_pth else None).to(device)
    if pre_train_pth:
        seg_model.load_from()

    # ====== Loss ======
    loss_rec_fn = CharbonnierLoss().to(device)
    loss_seg_fn = BceDiceLoss().to(device)

    optimizer = optim.AdamW(
        [
            {'params': ir_model.parameters(), 'lr': lr_ir},
            {'params': seg_model.parameters(), 'lr': lr_seg},
        ],
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    best_dice = 0.0
    best_path = os.path.join(ckpt_dir, 'best_joint_align.pth')
    no_improve = 0

    logger.info(f'联合训练 + 梯度促进 | lambda_seg={lambda_seg}, lambda_align={lambda_align}')

    # ================= 训练 =================
    for epoch in range(1, num_epochs + 1):
        ir_model.train()
        seg_model.train()

        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)

        for img_lq, img_hr, mask, _ in pbar:
            img_lq = img_lq.to(device)
            img_hr = img_hr.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            # ====== forward ======
            x_hat = ir_model(img_lq)[-1]
            logits = seg_model(x_hat)

            l_rec = loss_rec_fn(x_hat, img_hr)
            l_seg = loss_seg_fn(logits, mask)

            # ====== ⭐ 梯度促进核心 ======
            g_rec = torch.autograd.grad(l_rec, x_hat, retain_graph=True)[0]
            g_seg = torch.autograd.grad(l_seg, x_hat, retain_graph=True)[0]

            cos_sim = torch.nn.functional.cosine_similarity(
                g_rec.view(g_rec.size(0), -1),
                g_seg.view(g_seg.size(0), -1),
                dim=1
            ).mean()

            # ====== 总损失 ======
            loss = l_rec + lambda_seg * l_seg - lambda_align * cos_sim

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(
                rec=f"{l_rec.item():.4f}",
                seg=f"{l_seg.item():.4f}",
                align=f"{cos_sim.item():.4f}"
            )

        total_loss /= len(train_loader)
        scheduler.step()

        # ================= 验证 =================
        ir_model.eval()
        seg_model.eval()

        val_dice = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for img_lq, img_hr, mask, _ in valid_loader:
                img_lq = img_lq.to(device)
                mask = mask.to(device)

                x_hat = ir_model(img_lq)[-1]
                logits = seg_model(x_hat)

                val_dice += calculate_batch_dice(logits, mask).mean().item()
                val_iou += calculate_batch_iou(logits, mask).mean().item()

        val_dice /= len(valid_loader)
        val_iou /= len(valid_loader)

        logger.info(
            f'Epoch {epoch:03d} | loss {total_loss:.4f} | '
            f'Dice {val_dice:.4f} IoU {val_iou:.4f}'
        )

        # ====== 保存 ======
        if val_dice > best_dice:
            best_dice = val_dice
            no_improve = 0

            torch.save({
                'ir': ir_model.state_dict(),
                'seg': seg_model.state_dict(),
                'epoch': epoch
            }, best_path)

            logger.info(f'✔ 保存最佳模型 Dice={best_dice:.4f}')
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info('早停触发')
                break

    logger.info(f'训练结束，最佳 Dice={best_dice:.4f}')


if __name__ == '__main__':
    main()
