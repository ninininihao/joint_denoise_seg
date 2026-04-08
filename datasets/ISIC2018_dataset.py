import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 在运行 python train.py 之前，先覆盖路径
import os

# 强制修改数据集路径
ROOT_DIR = '/kaggle/input/isic2018/ISIC2018'  # 或者确认后的实际路径

# ==========================================
# 可视化工具函数
# ==========================================
def denormalize(tensor):
    """撤销归一化，用于可视化输出"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0.0, 1.0)


def overlay_mask_on_image(image_np, mask_np, color=[1.0, 0.0, 0.0], alpha=0.4):
    """
    将 mask 作为半透明图层覆盖在原图上
    image_np: [H, W, 3] 原图 numpy 数组 (0~1)
    mask_np: [H, W] 二值化 mask (0 和 1)
    color: 覆盖的颜色 RGB (默认红色)
    alpha: 掩码透明度 (0为全透明，1为全不透明)
    """
    overlay = np.copy(image_np)
    for c in range(3):
        # 仅在 mask == 1 的区域进行颜色融合: (1 - alpha) * 原图 + alpha * 颜色
        overlay[:, :, c] = np.where(
            mask_np > 0,
            image_np[:, :, c] * (1 - alpha) + color[c] * alpha,
            image_np[:, :, c]
        )
    return overlay


def visualize_dataset(dataloader, num_samples=3, sigma=0.1):
    """
    从 Dataloader 中取出一个 Batch 并画出三列图：原图、加噪图、原图+掩码
    """
    batch = next(iter(dataloader))

    # 获取数据并限制可视化数量
    lq_imgs, hq_imgs, masks, names = batch
    print(lq_imgs.shape, hq_imgs.shape, masks.shape)


    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)  # 保证 2D 索引不报错

    for i in range(num_samples):
        # 1. 张量反归一化并转为 Numpy 数组 [H, W, C]
        hq_vis = denormalize(hq_imgs[i]).permute(1, 2, 0).cpu().numpy()
        lq_vis = denormalize(lq_imgs[i]).permute(1, 2, 0).cpu().numpy()

        # Mask 转为 Numpy 数组 [H, W] (去掉通道维度)
        mask_vis = masks[i].squeeze(0).cpu().numpy()

        # 2. 生成带浅色遮罩的图层 (此处使用红色遮罩，透明度 0.4)
        overlay_vis = overlay_mask_on_image(hq_vis, mask_vis, color=[1.0, 0.0, 0.0], alpha=0.4)

        # 3. 绘制第一列: 原图
        axes[i, 0].imshow(hq_vis)
        axes[i, 0].set_title(f"Original (Clean)\n{names[i]}")
        axes[i, 0].axis('off')

        # 4. 绘制第二列: 加噪图
        axes[i, 1].imshow(lq_vis)
        axes[i, 1].set_title(f"Noisy ($\sigma={sigma}$)")
        axes[i, 1].axis('off')

        # 5. 绘制第三列: 原图+掩码覆盖
        axes[i, 2].imshow(overlay_vis)
        axes[i, 2].set_title("Original + Mask Overlay")
        axes[i, 2].axis('off')

    plt.tight_layout()
    # 存为图片到服务器当前目录
    save_path = f'visualize_sigma_{sigma}.png'
    plt.show()
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至: {save_path}")
    plt.close()


class ISIC2018Dataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=256, sigma=0.1, mask_suffix='_segmentation.png'):
        """
        ISIC 2018 数据集加载器

        参数:
            root_dir (str): 数据集根目录, 例如 '/root/datasets/ISIC2018'
            split (str): 'train', 'val', 或 'test'
            image_size (int): 图像统一缩放的大小
            sigma (float): 高斯白噪声的标准差 (例如: 0.05, 0.1, 0.2, 0.3)
            mask_suffix (str): mask 文件的后缀。ISIC 数据集的 mask 通常带有 '_segmentation.png' 后缀。
                               如果您的 mask 和原图名字完全一样只是后缀是 png，请改为 '.png'
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.sigma = sigma
        self.mask_suffix = mask_suffix

        # 定义路径
        self.img_dir = os.path.join(root_dir, split, 'image')
        self.mask_dir = os.path.join(root_dir, split, 'mask')

        # 获取所有图片文件名 (过滤掉非图片文件)
        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # 1. 基础的几何变换 (图片和Mask都需要)
        self.resize = transforms.Resize((self.image_size, self.image_size), interpolation=Image.BILINEAR)
        self.resize_mask = transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)

        # 2. 转换为 Tensor (会将像素值从 0-255 映射到 0.0-1.0)
        self.to_tensor = transforms.ToTensor()

        # 3. 标准归一化 (ImageNet 标准)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.img_names)

    def add_gaussian_noise(self, img_tensor):
        """
        添加高斯白噪声
        参数:
            img_tensor: 形状为 [C, H, W]，值域在 [0.0, 1.0] 之间的张量
        返回:
            加噪并截断到 [0.0, 1.0] 的张量
        """
        if self.sigma <= 0:
            return img_tensor

        # 生成与图像相同形状的均值为0，标准差为1的标准正态分布噪声，然后乘以 sigma
        noise = torch.randn_like(img_tensor) * self.sigma

        # 叠加噪声
        noisy_img = img_tensor + noise

        # 截断像素值，防止超出有效范围 [0.0, 1.0]
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)

        return noisy_img

    def __getitem__(self, idx):
        # 1. 获取文件名并读取图片
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 推导 Mask 文件名 (假设图片是 ISIC_0009954.jpg, mask 通常是 ISIC_0009954_segmentation.png)
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + self.mask_suffix
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 读取图像和掩码
        image = Image.open(img_path).convert('RGB')
        # 如果你的测试集没有mask文件，这里需要加个 try-except 或判断
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')  # 转换为灰度图
        else:
            # 如果没有Mask (比如在test集中)，创建一个全零的占位Mask
            mask = Image.new('L', image.size, 0)

        # 2. 调整大小 Resize
        image = self.resize(image)
        mask = self.resize_mask(mask)

        # 3. 转换为 Tensor (值域变为 0~1)
        image_tensor = self.to_tensor(image)
        mask_tensor = self.to_tensor(mask)

        # 二值化 Mask (确保 Mask 只有 0 和 1)
        mask_tensor = torch.where(mask_tensor > 0.5, torch.tensor(1.0), torch.tensor(0.0))

        # 4. 生成低质量图片 (添加噪声) -> 注意：在归一化之前加噪声最符合物理规律
        lq_image_tensor = self.add_gaussian_noise(image_tensor)

        # 5. 分别对 高质量原图 和 低质量噪图 进行标准归一化
        hq_image_norm = self.normalize(image_tensor)
        lq_image_norm = self.normalize(lq_image_tensor)

        # 返回字典形式，方便后续解包
        return lq_image_norm, hq_image_norm, mask_tensor, img_name



# ==========================================
# 使用示例与可视化 (方便你验证代码)
# ==========================================
if __name__ == '__main__':
    # ！！！请修改为你的实际路径 ！！！
    # ROOT_DIR = './datasets/ISIC2018'

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 数据集就在脚本所在目录下的 ISIC2018 文件夹里
    ROOT_DIR = os.path.join(current_dir, 'ISIC2018')

    # ！！！请根据你的 mask 文件夹里面的文件后缀修改 ！！！
    # 比如 mask 是 ISIC_0009954_segmentation.png，就填 '_segmentation.png'


    # 我们测试不同的 sigma: 0.05, 0.1, 0.2, 0.3
    test_sigmas = [0.05, 0.2]

    for sigma_val in test_sigmas:
        print(f"正在生成 sigma = {sigma_val} 的数据集...")

        dataset = ISIC2018Dataset(
            root_dir=ROOT_DIR,
            split='train',
            image_size=256,
            sigma=sigma_val
        )

        # 加载器
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # 调用可视化函数，画出前 3 个样本
        visualize_dataset(dataloader, num_samples=3, sigma=sigma_val)
