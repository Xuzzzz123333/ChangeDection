import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        attn_dim=64,
        num_heads=4,
        window_size=5,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ):
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError(
                f"SlidingWindowCrossAttention expects an odd window_size, got {window_size}."
            )
        if attn_dim % num_heads != 0:
            raise ValueError(
                f"attn_dim ({attn_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.window_size = window_size
        self.pad = window_size // 2
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(dim, attn_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim, attn_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, attn_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(attn_dim, dim, kernel_size=1, bias=False)

        self.unfold = nn.Unfold(kernel_size=window_size, padding=0)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def _pad_for_unfold(self, x):
        if self.pad == 0:
            return x
        pad = (self.pad, self.pad, self.pad, self.pad)
        pad_mode = "reflect" if min(x.shape[-2:]) > self.pad else "replicate"
        return F.pad(x, pad, mode=pad_mode)

    def forward(self, q_in, kv_in):
        b, _, h, w = q_in.shape
        hw = h * w
        window_area = self.window_size * self.window_size

        q = self.q_proj(q_in)
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)

        q = q.view(b, self.num_heads, self.head_dim, hw).permute(0, 1, 3, 2)
        q = F.normalize(q, dim=-1)

        # Reflect padding keeps local windows from seeing artificial zeros near borders.
        k = self._pad_for_unfold(k)
        v = self._pad_for_unfold(v)
        k = self.unfold(k).view(
            b, self.num_heads, self.head_dim, window_area, hw
        ).permute(0, 1, 4, 3, 2)
        v = self.unfold(v).view(
            b, self.num_heads, self.head_dim, window_area, hw
        ).permute(0, 1, 4, 3, 2)
        k = F.normalize(k, dim=-1)

        # q/k are already normalized, so using cosine-style logits without
        # extra 1/sqrt(d) scaling keeps the local attention from becoming too flat.
        attn = (q.unsqueeze(-2) * k).sum(dim=-1)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn.unsqueeze(-1) * v).sum(dim=-2)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, self.attn_dim, h, w)
        out = self.out_proj(out)
        return self.proj_drop(out)


class LightCrossGateFusion(nn.Module):
    def __init__(
        self,
        dim,
        attn_dim=64,
        num_heads=4,
        window_size=5,
        gamma_init=0.1,
    ):
        super().__init__()
        self.cross_attn = SlidingWindowCrossAttention(
            dim=dim,
            attn_dim=attn_dim,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gate = nn.Sequential(nn.Conv2d(2 * dim, dim, 1, bias=True), nn.Sigmoid())
        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x_high, x_low):
        x_high = F.interpolate(
            x_high, size=x_low.shape[-2:], mode="bilinear", align_corners=False
        )
        x_context = self.cross_attn(q_in=x_low, kv_in=x_high)
        x_high_guided = x_high + self.gamma * x_context

        gate = self.gate(torch.cat([x_high_guided, x_low], dim=1))
        fused = x_low + gate * x_high_guided
        return self.mix(fused)
