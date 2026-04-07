import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import calculate_batch_iou, calculate_batch_dice, BceDiceLoss
from models.SegModel.VMUnet import VMUNet
from models.IrModel.ConvIR import build_net
# 导入你定义的 Dataset 和指标函数
from datasets.ISIC2018_dataset import ISIC2018Dataset


# ==============================================================================
# 0. 基础设置与工具函数
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("SegmentationTraining")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ==============================================================================
# 2. 可视化函数 (三列：原图、真值叠加、预测叠加)
# ==============================================================================
def save_seg_visualization(img, gt, pred, epoch, save_dir):
    """
    可视化：[原图] | [GT Overlay] | [Pred Overlay]
    """
    os.makedirs(save_dir, exist_ok=True)
    idx = random.randint(0, img.size(0) - 1)

    # 转为 numpy [H, W]
    image_np = img[idx, 0].detach().cpu().numpy()
    gt_np = gt[idx, 0].detach().cpu().numpy()
    pred_np = torch.sigmoid(pred[idx, 0]).detach().cpu().numpy()
    pred_mask = (pred_np > 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 第一列：纯原始图像
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 第二列：真值覆盖 (浅蓝色)
    axes[1].imshow(image_np, cmap='gray')
    overlay_gt = np.zeros((*image_np.shape, 4))
    overlay_gt[gt_np > 0.5] = [0, 0.5, 1, 0.4]
    axes[1].imshow(overlay_gt)
    axes[1].set_title("GT Overlay")
    axes[1].axis('off')

    # 第三列：预测覆盖 (浅红色/珊瑚色)
    axes[2].imshow(image_np, cmap='gray')
    overlay_pred = np.zeros((*image_np.shape, 4))
    overlay_pred[pred_mask > 0.5] = [1, 0.4, 0.4, 0.4]
    axes[2].imshow(overlay_pred)
    axes[2].set_title("Pred Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'val_seg_epoch_{epoch:03d}.png'))
    plt.close()


# ==============================================================================
# 3. 主训练流程
# ==============================================================================
def main():
    # ---------- 超参数配置 ----------
    # data_path = "/root/datasets/ATLASv2_2D_Split811_MinPixel50/"
    data_path = "/home/liushasha/JointDenoiseSeg/datasets/ISIC2018"
    out_dir = "/home/liushasha/JointDenoiseSeg/results/train_ir_seg"
    # out_dir = "/root/code/FM/results/Seg/ConvIR_Seg(VMUnet)_ATLASv2"
    batch_size = 2
    num_epochs = 100
    lr = 1e-4
    seed = 42
    patience = 15  # 连续 15 轮不涨就停止
    no_optim_count = 0  # 当前连续不涨的轮数计数

    set_seed(seed)
    logger = get_logger(out_dir)
    vis_dir = os.path.join(out_dir, "visualizations")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"========== 开始分割任务训练 ==========")

    # ---------- 数据加载 ----------
    train_loader = DataLoader(ISIC2018Dataset(data_path, split='train', image_size=256, sigma=0.2), batch_size=batch_size,
                              shuffle=True, num_workers=8)
    valid_loader = DataLoader(ISIC2018Dataset(data_path, split='val', image_size=256, sigma=0.2), batch_size=batch_size,
                              shuffle=False, num_workers=8)
    test_loader = DataLoader(ISIC2018Dataset(data_path, split='test', image_size=256, sigma=0.2), batch_size=batch_size,
                             shuffle=False, num_workers=8)

    # ---------- 初始化模型、损失、优化器 ----------
    # --- 模型与优化器 ---
    # pre_train_pth = '/root/code/pre_trained/vmamba_small_e238_ema.pth'
    pre_train_pth = '/home/liushasha/JointDenoiseSeg/pretrained/vmamba_small_e238_ema.pth'
    # 要先训练去噪模型得到去噪的权重，将最好性能的去噪模型的权重地址放在下面
    ir_model_pth = '/home/liushasha/JointDenoiseSeg/results/train_ConvIR_ISIC2018_Denoise_Gaussian_sigma0.2/checkpoints/best_model.pth'
    ir_model = build_net(version='base', in_nc=3, out_nc=3).to(device)
    ir_model.load_state_dict(torch.load(ir_model_pth))
    for param in ir_model.parameters():
        param.requires_grad = False
    ir_model.eval()
    # 分割模型、加载分割模型的权重
    model = VMUNet(num_classes=1, load_ckpt_path=pre_train_pth).to(device)
    model.load_from()

    if model is None:
        logger.error("请先在代码中定义并实例化你的分割模型！")
        return

    criterion = BceDiceLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_dice = 0.0
    best_model_path = os.path.join(ckpt_dir, "best_seg_model.pth")

    # ---------- 训练循环 ----------
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch:03d}/{num_epochs:03d}] Train", leave=False)

        for img_lq, img_hr, mask, fname in train_pbar:
            # 验证恢复网络恢复出来的分割性能：通常使用 img_lq (低质量) 作为输入
            img_in = img_lq.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                img_ins = ir_model(img_in)
                img_in = img_ins[-1]

            optimizer.zero_grad()
            pred = model(img_in)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_loss /= len(train_loader)
        scheduler.step()

        # ---------- 验证阶段 ----------
        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        saved_vis = False

        with torch.no_grad():
            valid_pbar = tqdm(valid_loader, desc=f"Validating", leave=False)
            for img_lq, img_hr, mask, fname in valid_pbar:
                img_in = img_lq.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    img_ins = ir_model(img_in)
                    img_in = img_ins[-1]

                pred= model(img_in)
                loss = criterion(pred, mask)
                val_loss += loss.item()

                # 计算指标并累加 (Batch 内所有样本的均值再求和)
                batch_dice = calculate_batch_dice(pred, mask).mean()
                batch_iou = calculate_batch_iou(pred, mask).mean()
                val_dice += batch_dice.item()
                val_iou += batch_iou.item()

                if not saved_vis and random.random() > 0.8:
                    save_seg_visualization(img_in, mask, pred, epoch, vis_dir)
                    saved_vis = True

        val_loss /= len(valid_loader)
        val_dice /= len(valid_loader)
        val_iou /= len(valid_loader)

        logger.info(f"Epoch [{epoch:03d}] | Train Loss: {train_loss:.4f} | Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | mIoU: {val_iou:.4f}")

        # 保存最佳模型 (基于 Dice 系数)
        if val_dice > best_dice:
            best_dice = val_dice
            no_optim_count = 0  # 性能提升，重置计数
            torch.save(model.state_dict(), best_model_path)
            logger.info(f" ---> 发现新最佳模型! Dice: {best_dice:.4f}，已保存。")
        else:
            no_optim_count += 1  # 性能没提升，累加计数

        if no_optim_count >= patience:
            logger.info(f"==== 触发早停机制 ==== 连续 {patience} 轮验证集 Dice 未提升，停止训练。")
            break

    logger.info(f"训练完成。最佳 Dice: {best_dice:.4f}")

    # ==============================================================================
    # 5. 测试阶段 (使用训练中保存的最佳模型)
    # ==============================================================================
    logger.info("\n" + "=" * 50)
    logger.info("========== 开始在 Test 集上评估最佳模型 ==========")
    logger.info(f"加载最佳模型权重: {best_model_path}")

    # 重新加载最优权重
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_dice = 0.0
    test_iou = 0.0

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for img_lq, img_hr, mask, fname in test_pbar:
            img_in = img_lq.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                img_ins = ir_model(img_in)
                img_in = img_ins[-1]

            pred = model(img_in)

            # 计算该 Batch 的平均指标
            # 这里调用的是你提供的 calculate_batch 函数
            b_dice = calculate_batch_dice(pred, mask).mean()
            b_iou = calculate_batch_iou(pred, mask).mean()

            test_dice += b_dice.item()
            test_iou += b_iou.item()

    # 计算整个测试集的平均值
    test_dice /= len(test_loader)
    test_iou /= len(test_loader)

    logger.info(f"【最终测试性能】")
    logger.info(f"Test Dice: {test_dice:.4f}")
    logger.info(f"Test mIoU: {test_iou:.4f}")
    logger.info("==========================================")


if __name__ == "__main__":
    main()