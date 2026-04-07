from .vmamba import VSSM
import torch
from torch import nn


class VMUNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 2, 2],
                 depths_decoder=[2, 2, 2, 1],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           )

    def forward(self, x, return_feat=False):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if return_feat:
            logits, features = self.vmunet(x, return_feat=True)
            return logits, features
        else:
            logits = self.vmunet(x)
            return logits

    def load_from(self):
        if self.load_ckpt_path is not None:
            print(f"Loading weights from: {self.load_ckpt_path}")
            device = next(self.parameters()).device
            modelCheckpoint = torch.load(self.load_ckpt_path, map_location='cpu')

            if 'model' in modelCheckpoint:
                pretrained_dict = modelCheckpoint['model']
            else:
                pretrained_dict = modelCheckpoint

            model_dict = self.vmunet.state_dict()

            # --- 1. Encoder Loading (Strict Shape Match) ---
            encoder_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    encoder_dict[k] = v

            # 更新 Encoder
            model_dict.update(encoder_dict)
            print(f'Encoder loaded keys: {len(encoder_dict)} / {len(pretrained_dict)}')

            # --- 2. Decoder Loading (Symmetric Init with Shape Check) ---
            decoder_dict = {}
            for k, v in pretrained_dict.items():
                new_k = None
                # 映射逻辑
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')

                # 关键：检查 Key 存在 且 形状匹配
                if new_k and new_k in model_dict:
                    if model_dict[new_k].shape == v.shape:
                        decoder_dict[new_k] = v

            # 更新 Decoder
            model_dict.update(decoder_dict)
            print(f'Decoder loaded keys: {len(decoder_dict)}')

            self.vmunet.load_state_dict(model_dict)
            print("Weights loading finished!")


if __name__ == '__main__':
    a = torch.randn((1, 1, 192, 224), device=torch.device('cuda'))
    model = VMUNet(load_ckpt_path='/root/mamba_package/vmamba_small_e238_ema.pth').cuda()
    model.load_from()
    y = model(a)
    print(y.shape)

