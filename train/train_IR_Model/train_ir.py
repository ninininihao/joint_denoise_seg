# 在训练脚本的最开头添加
data_path = "/home/liushasha/JointDenoiseSeg/datasets/ISIC2018"
import torch
import gc

# 清空缓存
torch.cuda.empty_cache()
gc.collect()

# 设置内存分配
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

# import os
# # 将这里的路径替换为你的实际路径
# cudnn_path = r"C:\anaconda3\envs\mamba\Library\bin"
# os.environ['PATH'] = cudnn_path + os.pathsep + os.environ.get('PATH', '')
import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from datasets.ISIC2018_dataset import ISIC2018Dataset
from models.IrModel.ConvIR import build_net
from tqdm import tqdm


# ==============================================================================
# 0. 基础设置与工具函数
# ==============================================================================
def set_seed(seed=42):
    """设置固定的随机种子以保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_dir):
    """配置日志记录器，同时输出到控制台和文件"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("DenoisingTraining")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件 Handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setFormatter(formatter)

    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def calculate_metrics(pred, gt):
    """
    使用业界标准 (skimage) 计算 PSNR 和 SSIM。
    假设输入为 [B, C, H, W] 的 Tensor，并转为 numpy 计算。
    """
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()

    batch_size = pred_np.shape[0]
    psnr_val = 0.0
    ssim_val = 0.0

    for i in range(batch_size):
        p = pred_np[i, 0]  # 提取单通道 [H, W]
        g = gt_np[i, 0]

        # 确定数据范围 (根据 Ground Truth 的最大最小值)
        data_range = g.max() - g.min()
        if data_range == 0:
            data_range = 1.0  # 防止除零

        psnr_val += compute_psnr(g, p, data_range=data_range)
        ssim_val += compute_ssim(g, p, data_range=data_range)

    return psnr_val / batch_size, ssim_val / batch_size


def save_validation_visualization(gt, pred, epoch, save_dir):
    """验证时随机保存一个 batch 中的一张图像的对比结果（两列：真值、去噪预测）"""
    os.makedirs(save_dir, exist_ok=True)

    # 随机选 batch 中的一张
    idx = random.randint(0, gt.size(0) - 1)
    g = gt[idx, 0].detach().cpu().numpy()
    p = pred[idx, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(g, cmap='gray')
    axes[0].set_title("Ground Truth (Clean)")
    axes[0].axis('off')

    axes[1].imshow(p, cmap='gray')
    axes[1].set_title("Prediction (Denoised)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'val_vis_epoch_{epoch:03d}.png'))
    plt.close()

# ==============================================================================
# 2. 损失函数代码
# ==============================================================================
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (SwinIR 官方推荐用于图像恢复)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


# ==============================================================================
# 4. 主训练流程
# ==============================================================================
def main():
    # ---------- 超参数配置 ----------
    # data_path = "/root/datasets/ATLASv2_2D_Split811_MinPixel50/"  # 数据集路径
    # data_path = "/datasets/ISIC2018"  # 数据集路径
    # data_path = "../datasets/ISIC2018"
    # out_dir = "../results/train_ConvIR_ISIC2018_Denoise_Gaussian_sigma0.2"  # 输出目录
    data_path = "/home/liushasha/JointDenoiseSeg/datasets/ISIC2018"
    print(f"DEBUG: data_path = {data_path}")
    out_dir = "/home/liushasha/JointDenoiseSeg/results/train_ConvIR_ISIC2018_Denoise_Gaussian_sigma0.2"
    noise_level = 9.0  # 噪声比例
    batch_size = 2  # 批次大小
    num_epochs = 100  # 训练轮数
    lr = 2e-4  # 初始学习率 (去噪推荐 1e-4 到 2e-4)
    seed = 42  # 随机种子
    patience = 15  # 连续 15 轮不涨就停止
    no_optim_count = 0  # 当前连续不涨的轮数计数

    # 设置种子、日志和输出路径
    set_seed(seed)
    logger = get_logger(out_dir)
    vis_dir = os.path.join(out_dir, "visualizations")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info(f"========== 开始去噪任务训练 ==========")
    logger.info(f"随机种子: {seed}, 噪声水平: {noise_level}%, 批大小: {batch_size}, 初始学习率：{lr}")
    print("==== 使用的数据路径 ====", data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用计算设备: {device}")

    # ---------- 数据加载 ----------
    try:
        # train_dataset = ATLAS2DDataset(data_path, split='train', noise_level=noise_level)
        # valid_dataset = ATLAS2DDataset(data_path, split='valid', noise_level=noise_level)
        # test_dataset = ATLAS2DDataset(data_path, split='test', noise_level=noise_level)
        train_dataset = ISIC2018Dataset(data_path, split='train', sigma=0.2)
        valid_dataset = ISIC2018Dataset(data_path, split='val', sigma=0.2)
        test_dataset = ISIC2018Dataset(data_path, split='test', sigma=0.2)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    except FileNotFoundError:
        logger.error(f"找不到数据集，请检查路径: {data_path}。")
        return

    # ---------- 初始化模型、损失、优化器 ----------
    model = build_net(version='base', in_nc=3, out_nc=3).to(device)
    # model = AdaIR(
    #     inp_channels=1,
    #     out_channels=1,
    #     dim=48,
    #     decoder=True
    # ).to(device)


    criterion = CharbonnierLoss().to(device)

    # 推荐使用 AdamW 优化器，对恢复任务效果更好
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # 使用余弦退火学习率，平滑降低学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_psnr = 0.0
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")

    # ---------- 训练循环 ----------
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        # 【修改点 1】：使用 tqdm 包装 train_loader
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch:03d}/{num_epochs:03d}] Train", leave=False)

        for batch_idx, (img_lq, img_hr, mask, fname) in enumerate(train_pbar):
            img_lq = img_lq.to(device)
            img_hr = img_hr.to(device)

            optimizer.zero_grad()
            preds = model(img_lq)
            pred = preds[-1]
            loss = criterion(pred, img_hr)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 实时在进度条后缀显示当前的 Loss
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_loss /= len(train_loader)
        scheduler.step()

        # ---------- 验证阶段 ----------
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        saved_vis = False

        with torch.no_grad():
            # 【修改点 2】：使用 tqdm 包装 valid_loader
            valid_pbar = tqdm(valid_loader, desc=f"Epoch [{epoch:03d}/{num_epochs:03d}] Valid", leave=False)

            for img_lq, img_hr, mask, fname in valid_pbar:
                img_lq = img_lq.to(device)
                img_hr = img_hr.to(device)

                preds = model(img_lq)
                pred = preds[-1]
                loss = criterion(pred, img_hr)
                val_loss += loss.item()

                # 计算指标
                b_psnr, b_ssim = calculate_metrics(pred, img_hr)
                val_psnr += b_psnr
                val_ssim += b_ssim

                # 随机保存当前 epoch 的一组可视化图
                if not saved_vis and random.random() > 0.5:
                    save_validation_visualization(img_hr, pred, epoch, vis_dir)
                    saved_vis = True

            # 如果跑完了还没保存，强制保存最后一次 batch 的第一张
            if not saved_vis:
                save_validation_visualization(img_hr, pred, epoch, vis_dir)

        val_loss /= len(valid_loader)
        val_psnr /= len(valid_loader)
        val_ssim /= len(valid_loader)

        # 这一行 logger 输出会保留在控制台，代表这一轮跑完了
        logger.info(f"Epoch [{epoch:03d}/{num_epochs:03d}] | LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")

        # 保存最佳模型权重 (基于 PSNR)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            no_optim_count = 0  # 性能提升，重置计数
            torch.save(model.state_dict(), best_model_path)
            logger.info(f" ---> 发现新最佳模型! 当前最高 PSNR: {best_psnr:.2f} dB，已保存。")
        else:
            no_optim_count += 1  # 性能没提升，累加计数

        if no_optim_count >= patience:
            logger.info(f"==== 触发早停机制 ==== 连续 {patience} 轮验证集 Dice 未提升，停止训练。")
            break

    logger.info("========== 训练结束 ==========")

    # ==============================================================================
    # 5. 测试阶段 (Test)
    # ==============================================================================
    logger.info("========== 开始在 Test 集上评估最佳模型 ==========")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_psnr = 0.0
    test_ssim = 0.0

    with torch.no_grad():
        # 【修改点 3】：使用 tqdm 包装 test_loader
        test_pbar = tqdm(test_loader, desc="Testing")
        for img_lq, img_hr, mask, fname in test_pbar:
            img_lq = img_lq.to(device)
            img_hr = img_hr.to(device)

            preds = model(img_lq)
            pred = preds[-1]
            b_psnr, b_ssim = calculate_metrics(pred, img_hr)
            test_psnr += b_psnr
            test_ssim += b_ssim

    test_psnr /= len(test_loader)
    test_ssim /= len(test_loader)

    logger.info(f"【最终测试性能】")
    logger.info(f"最优权重路径: {best_model_path}")
    logger.info(f"Test PSNR: {test_psnr:.2f} dB")
    logger.info(f"Test SSIM: {test_ssim:.4f}")
    logger.info("==========================================")


if __name__ == "__main__":
    main()