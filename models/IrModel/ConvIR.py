from .layers import *  # 请确保你的 layers.py 中包含 ResBlock 和 BasicConv


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res - 1)]
        layers.append(ResBlock(out_channel, out_channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()
        layers = [ResBlock(channel, channel) for _ in range(num_res - 1)]
        layers.append(ResBlock(channel, channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, out_plane, in_nc=1):
        super(SCM, self).__init__()
        # 改造点 1：将输入通道从 3 改为 in_nc (默认为 1)
        self.main = nn.Sequential(
            BasicConv(in_nc, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        return self.main(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel * 2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


class ConvIR(nn.Module):
    def __init__(self, version='base', in_nc=1, out_nc=1):
        super(ConvIR, self).__init__()

        # 确定残差块数量
        if version == 'small':
            num_res = 4
        elif version == 'base':
            num_res = 8
        elif version == 'large':
            num_res = 16
        else:
            num_res = 8

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])

        # 改造点 2：适配输入通道 in_nc 和输出通道 out_nc
        self.feat_extract = nn.ModuleList([
            BasicConv(in_nc, base_channel, kernel_size=3, relu=True, stride=1),  # 第 0 层：输入
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, out_nc, kernel_size=3, relu=False, stride=1)  # 第 5 层：最终输出
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        # 改造点 3：多尺度输出头的通道数改为 out_nc
        self.ConvsOut = nn.ModuleList([
            BasicConv(base_channel * 4, out_nc, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, out_nc, kernel_size=3, relu=False, stride=1),
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4, in_nc=in_nc)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2, in_nc=in_nc)

    def forward(self, x):
        # 多尺度辅助输入
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        # Encoder 路径
        # 1/1 尺度 (256x256)
        x_feat = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_feat)

        # 1/2 尺度 (128x128)
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        # 1/4 尺度 (64x64)
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        # Decoder 路径
        # 处理 1/4 尺度并输出残差
        z = self.Decoder[0](z)
        z_out4 = self.ConvsOut[0](z)
        outputs.append(z_out4 + x_4)  # 尺度 1/4 的输出

        # 上采样至 1/2 尺度
        z = self.feat_extract[3](z)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_out2 = self.ConvsOut[1](z)
        outputs.append(z_out2 + x_2)  # 尺度 1/2 的输出

        # 上采样至 1/1 尺度
        z = self.feat_extract[4](z)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z_final = self.feat_extract[5](z)
        outputs.append(z_final + x)  # 最终全尺寸输出

        return outputs


def build_net(version='base', in_nc=1, out_nc=1):
    return ConvIR(version=version, in_nc=in_nc, out_nc=out_nc)

if __name__ == '__main__':
    # 1. 初始化模型 (假设使用 Base 版本，适配单通道 MRI)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_net(version='base', in_nc=1, out_nc=1).to(device)
    model.eval()

    # 2. 模拟输入数据: [Batch_size, Channel, Height, Width]
    # 模拟一个 Batch 为 4 的 256x256 单通道 MRI 切片
    dummy_input = torch.randn(4, 1, 192, 224).to(device)

    print(f"输入维度: {dummy_input.shape}")
    print("-" * 30)

    # 3. 前向传播
    with torch.no_grad():
        outputs = model(dummy_input)

    # 4. 打印每一层输出的维度
    # 模型会返回一个列表，包含从低分辨率到高分辨率的 3 个结果
    scales = ["1/4 尺度 (Smallest)", "1/2 尺度 (Medium)", "1/1 尺度 (Final Full)"]

    for i, out in enumerate(outputs):
        print(f"输出层 {i + 1} [{scales[i]}]:")
        print(f"   维度: {out.shape}")
        print(f"   数值范围约: [{out.min().item():.2f}, {out.max().item():.2f}]")
        print("-" * 30)

    # 5. 提示：如何取最终降噪结果
    final_denoised_image = outputs[-1]
    print(f"1/1 图像维度为 {final_denoised_image.shape}")