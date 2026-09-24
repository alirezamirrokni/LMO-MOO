from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional


class MultiMNISTViT(nn.Module):
    """
    Two-task ViT for MultiMNIST-like setting.
    Input:  (B, C, H=28, W=43)  PNG tensor
    Output: logits1 (B, 10), logits2 (B, 10)
    """

    def __init__(
        self,
        img_h: int = 28,
        img_w: int = 43,
        in_channels: int = 1,
        patch_size: int = 4,
        emb_size: int = 16,
        nhead: int = 2,
        num_layers: int = 3,
        num_classes_task1: int = 10,
        num_classes_task2: int = 10,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.emb_size = emb_size

        # --- padding to make (H, W) divisible by patch_size ---
        self.pad_h = (patch_size - (img_h % patch_size)) % patch_size
        self.pad_w = (patch_size - (img_w % patch_size)) % patch_size

        padded_h = img_h + self.pad_h
        padded_w = img_w + self.pad_w

        self.grid_h = padded_h // patch_size
        self.grid_w = padded_w // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # --- patch projection (conv -> linear) ---
        self.patch_dim = (patch_size ** 2) * in_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.patch_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.patch_emb = nn.Linear(self.patch_dim, emb_size)

        # --- [CLS] token + positional embedding ---
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_size))
        self.pos_emb = nn.Parameter(torch.rand(1, self.num_patches + 1, emb_size))

        if dim_feedforward is None:
            dim_feedforward = emb_size * 4

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # --- two task heads (命名保持不变) ---
        self.head1 = nn.Linear(emb_size, num_classes_task1)
        self.head2 = nn.Linear(emb_size, num_classes_task2)

    # =========================
    # 关键：补齐 trainer 需要的接口
    # =========================
    def shared_parameters(self):
        """
        Return parameters shared by all tasks.
        Used by FAMO trainer to cache last_shared_parameters.
        """
        params = []
        params += list(self.conv.parameters())
        params += list(self.patch_emb.parameters())
        params += [self.cls_token, self.pos_emb]
        params += list(self.transformer_enc.parameters())
        return params

    def task_specific_parameters(self):
        """
        Return task-specific head parameters.
        Many MTL weight methods need this interface.
        """
        return list(self.head1.parameters()) + list(self.head2.parameters())

    def forward(self, x: torch.Tensor, return_representation: bool = False):
        """
        x: (B, C, H, W) expected H=28, W=43 (can be padded internally)
        returns:
          - if return_representation=False: (logits1, logits2)
          - if return_representation=True : ((logits1, logits2), rep)
            where rep is CLS embedding with shape (B, emb_size)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape (B,C,H,W), got {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got C={x.size(1)}")

        if self.pad_w != 0 or self.pad_h != 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="constant", value=0.0)

        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_emb(x)

        cls = self.cls_token.expand(x.size(0), 1, self.emb_size)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_emb

        y = self.transformer_enc(x)

        rep = y[:, 0, :]          # (B, emb_size)
        logits1 = self.head1(rep) # (B, 10)
        logits2 = self.head2(rep) # (B, 10)

        preds = (logits1, logits2)

        if return_representation:
            return preds, rep
        return preds


if __name__ == "__main__":
    model = MultiMNISTViT(img_h=28, img_w=43, in_channels=1, patch_size=4, emb_size=16)
    x = torch.rand(5, 1, 28, 43)
    y1, y2 = model(x)
    print(y1.shape, y2.shape)
