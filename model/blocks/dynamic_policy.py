import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from dinov3.utils import cat_keep_shapes, uncat_with_shapes


DEFAULT_POLICY_INIT_KEEP_PROB = 0.98


def safe_inverse_sigmoid(prob: float, eps: float = 1e-6) -> float:
    prob = min(max(float(prob), eps), 1.0 - eps)
    return math.log(prob / (1.0 - prob))


def infer_num_prefix_tokens(model, override: Optional[int] = None) -> int:
    if override is not None and int(override) >= 0:
        return int(override)
    return 1 + int(getattr(model, "n_storage_tokens", 0))


def sample_policy(
    logits: torch.Tensor,
    mode: str = "soft",
    temperature: float = 1.0,
    hard: bool = False,
    threshold: float = 0.5,
    training: bool = True,
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    if mode not in {"soft", "gumbel"}:
        raise ValueError(f"Unsupported policy mode: {mode}")

    if mode == "gumbel" and training:
        uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
        logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
        y_soft = torch.sigmoid((logits + logistic_noise) / temperature)
    else:
        y_soft = torch.sigmoid(logits / temperature)

    if not hard:
        return y_soft

    y_hard = (y_soft > threshold).to(dtype=y_soft.dtype)
    if training:
        return y_hard.detach() - y_soft.detach() + y_soft
    return y_hard


def apply_topk_policy(policy: torch.Tensor, keep_ratio: Optional[float]) -> torch.Tensor:
    if keep_ratio is None:
        return policy
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("head_topk_ratio must be in (0, 1].")

    num_items = policy.shape[-1]
    k = max(int(round(num_items * keep_ratio)), 1)
    if k >= num_items:
        return torch.ones_like(policy)

    topk_indices = torch.topk(policy, k=k, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(policy)
    mask.scatter_(-1, topk_indices, 1.0)
    return mask


class DynamicPolicyNet(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_patch_tokens: int,
        use_head_policy: bool = True,
        use_block_policy: bool = False,
        use_token_policy: bool = False,
        init_keep_prob: float = DEFAULT_POLICY_INIT_KEEP_PROB,
    ):
        super().__init__()
        hidden_dim = max(int(hidden_dim), 1)
        self.num_heads = int(num_heads)
        self.num_patch_tokens = int(num_patch_tokens)
        self.use_head_policy = bool(use_head_policy)
        self.use_block_policy = bool(use_block_policy)
        self.use_token_policy = bool(use_token_policy)

        self.fc1 = nn.Linear(embed_dim * 4, hidden_dim)
        self.act = nn.GELU()
        self.head_head = nn.Linear(hidden_dim, self.num_heads) if self.use_head_policy else None
        self.block_head = nn.Linear(hidden_dim, 1) if self.use_block_policy else None
        self.token_head = nn.Linear(hidden_dim, self.num_patch_tokens) if self.use_token_policy else None
        self.reset_parameters(init_keep_prob=init_keep_prob)

    def reset_parameters(self, init_keep_prob: float = DEFAULT_POLICY_INIT_KEEP_PROB):
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        init_bias = safe_inverse_sigmoid(init_keep_prob)
        for head in (self.head_head, self.block_head, self.token_head):
            if head is None:
                continue
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, init_bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, num_prefix_tokens: int) -> Dict[str, Optional[torch.Tensor]]:
        if x1.shape != x2.shape:
            raise ValueError(f"Expected matching token shapes, got {x1.shape} vs {x2.shape}.")
        if x1.ndim != 3:
            raise ValueError(f"Expected [B, N, C] tokens, got {x1.shape}.")
        if num_prefix_tokens < 0 or num_prefix_tokens >= x1.shape[1]:
            raise ValueError(
                f"num_prefix_tokens must be in [0, N-1], got {num_prefix_tokens} for N={x1.shape[1]}."
            )

        x1_patch = x1[:, num_prefix_tokens:]
        x2_patch = x2[:, num_prefix_tokens:]
        p1 = x1_patch.mean(dim=1)
        p2 = x2_patch.mean(dim=1)
        pd = torch.abs(x1_patch - x2_patch).mean(dim=1)
        pm = (x1_patch * x2_patch).mean(dim=1)

        rel = torch.cat([p1, p2, pd, pm], dim=-1)
        hidden = self.act(self.fc1(rel))
        return {
            "head_logits": self.head_head(hidden) if self.head_head is not None else None,
            "block_logits": self.block_head(hidden) if self.block_head is not None else None,
            "token_logits": self.token_head(hidden) if self.token_head is not None else None,
        }


class PolicyAwareSelfAttention(nn.Module):
    def __init__(self, attn: nn.Module, layer_index: int):
        super().__init__()
        self.layer_index = int(layer_index)
        self.num_heads = int(attn.num_heads)
        self.scale = getattr(attn, "scale", None)
        self.qkv = attn.qkv
        self.attn_drop = getattr(attn, "attn_drop", nn.Identity())
        self.proj = attn.proj
        self.proj_drop = getattr(attn, "proj_drop", nn.Identity())
        self._runtime_head_policy = None

    @staticmethod
    def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    @classmethod
    def rope_apply(cls, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (cls.rope_rotate_half(x) * sin)

    def apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, rope: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        n_tokens = q.shape[-2]
        prefix = n_tokens - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = self.rope_apply(q[:, :, prefix:, :], sin, cos)
        q = torch.cat((q_prefix, q), dim=-2)
        k_prefix = k[:, :, :prefix, :]
        k = self.rope_apply(k[:, :, prefix:, :], sin, cos)
        k = torch.cat((k_prefix, k), dim=-2)
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def set_runtime_head_policy(self, head_policy: Optional[torch.Tensor]):
        self._runtime_head_policy = head_policy

    def clear_runtime_head_policy(self):
        self._runtime_head_policy = None

    def compute_attention(
        self,
        qkv: torch.Tensor,
        attn_bias=None,
        rope=None,
        head_policy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_tokens, _ = qkv.shape
        dim = self.qkv.in_features

        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, dim // self.num_heads)
        q, k, v = torch.unbind(qkv, dim=2)
        q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        attn_kwargs = {}
        if attn_bias is not None:
            if not torch.is_tensor(attn_bias):
                raise ValueError(
                    f"Unsupported attn_bias type at layer {self.layer_index}: "
                    f"{type(attn_bias)!r}. Expected a Tensor compatible with "
                    "torch.nn.functional.scaled_dot_product_attention(attn_mask=...)."
                )
            attn_kwargs["attn_mask"] = attn_bias
        dropout_p = float(getattr(self.attn_drop, "p", 0.0)) if self.training else 0.0
        try:
            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_p,
                **attn_kwargs,
            )
        except (RuntimeError, TypeError) as exc:
            raise ValueError(
                f"Failed to apply scaled_dot_product_attention at layer {self.layer_index}. "
                f"attn_bias type={type(attn_bias)!r}"
            ) from exc
        x = x.transpose(1, 2)

        active_policy = self._runtime_head_policy if head_policy is None else head_policy
        if active_policy is not None:
            if active_policy.shape != (batch_size, self.num_heads):
                raise ValueError(
                    f"Runtime head policy shape mismatch at layer {self.layer_index}: "
                    f"expected {(batch_size, self.num_heads)}, got {tuple(active_policy.shape)}."
                )
            x = x * active_policy[:, None, :, None].to(dtype=x.dtype)
        return x.reshape(batch_size, num_tokens, dim)

    def forward(self, x: torch.Tensor, attn_bias=None, rope=None) -> torch.Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list: List[torch.Tensor], attn_bias=None, rope_list=None) -> List[torch.Tensor]:
        if rope_list is None:
            rope_list = [None] * len(x_list)
        assert len(x_list) == len(rope_list)
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for qkv, _, rope in zip(qkv_list, shapes, rope_list):
            att_out.append(self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        x_flat = self.proj_drop(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)
