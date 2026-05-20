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
        policy_granularity: str = "image",
    ):
        super().__init__()
        hidden_dim = max(int(hidden_dim), 1)
        self.num_heads = int(num_heads)
        self.num_patch_tokens = int(num_patch_tokens)
        self.use_head_policy = bool(use_head_policy)
        self.use_block_policy = bool(use_block_policy)
        self.use_token_policy = bool(use_token_policy)
        self.policy_granularity = str(policy_granularity)

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

        if self.policy_granularity == "token":
            # Per-token: compute relation features at each spatial position
            # rel shape: [B, N_patch, 4*C]
            rel = torch.cat([
                x1_patch,
                x2_patch,
                torch.abs(x1_patch - x2_patch),
                x1_patch * x2_patch,
            ], dim=-1)
            # hidden: [B, N_patch, hidden_dim]
            hidden = self.act(self.fc1(rel))
            # head_logits: [B, N_patch, num_heads]
            head_logits = self.head_head(hidden) if self.head_head is not None else None
            # block/token logits still use pooled (global decision)
            hidden_pooled = hidden.mean(dim=1)
            block_logits = self.block_head(hidden_pooled) if self.block_head is not None else None
            token_logits = self.token_head(hidden_pooled) if self.token_head is not None else None
        else:
            # Per-image (AdaViT default): pool then predict
            p1 = x1_patch.mean(dim=1)
            p2 = x2_patch.mean(dim=1)
            pd = torch.abs(x1_patch - x2_patch).mean(dim=1)
            pm = (x1_patch * x2_patch).mean(dim=1)
            rel = torch.cat([p1, p2, pd, pm], dim=-1)
            hidden = self.act(self.fc1(rel))
            head_logits = self.head_head(hidden) if self.head_head is not None else None
            block_logits = self.block_head(hidden) if self.block_head is not None else None
            token_logits = self.token_head(hidden) if self.token_head is not None else None

        return {
            "head_logits": head_logits,
            "block_logits": block_logits,
            "token_logits": token_logits,
        }


class PolicyAwareMlp(nn.Module):
    def __init__(self, mlp: nn.Module, layer_index: int, num_heads: int):
        super().__init__()
        self.layer_index = int(layer_index)
        self.num_heads = int(num_heads)
        self._runtime_head_policy = None

        if hasattr(mlp, "fc1") and hasattr(mlp, "fc2"):
            self.impl_type = "mlp"
            self.fc1 = mlp.fc1
            self.act = mlp.act
            self.fc2 = mlp.fc2
            self.drop = mlp.drop
        elif hasattr(mlp, "w1") and hasattr(mlp, "w2") and hasattr(mlp, "w3"):
            self.impl_type = "swiglu"
            self.w1 = mlp.w1
            self.w2 = mlp.w2
            self.w3 = mlp.w3
        else:
            raise TypeError(
                f"Unsupported MLP module for policy gating at layer {self.layer_index}: "
                f"{type(mlp).__name__}"
            )

    def set_runtime_head_policy(self, head_policy: Optional[torch.Tensor]):
        self._runtime_head_policy = head_policy

    def clear_runtime_head_policy(self):
        self._runtime_head_policy = None

    def _apply_input_gate(self, x: torch.Tensor) -> torch.Tensor:
        active_policy = self._runtime_head_policy
        if active_policy is None:
            return x
        batch_size, num_tokens, dim = x.shape
        if dim % self.num_heads != 0:
            raise ValueError(
                f"MLP input dim {dim} is not divisible by num_heads={self.num_heads} "
                f"at layer {self.layer_index}."
            )
        head_dim = dim // self.num_heads
        if active_policy.ndim == 2:
            # per-image: [B, H] → [B, 1, dim]
            if active_policy.shape != (batch_size, self.num_heads):
                raise ValueError(
                    f"Runtime MLP head policy shape mismatch at layer {self.layer_index}: "
                    f"expected {(batch_size, self.num_heads)}, got {tuple(active_policy.shape)}."
                )
            gate = active_policy[:, :, None].expand(-1, -1, head_dim).reshape(batch_size, 1, dim)
        elif active_policy.ndim == 3:
            # per-token: [B, N, H] → [B, N, dim]
            gate = active_policy[:, :, :, None].expand(-1, -1, -1, head_dim).reshape(batch_size, num_tokens, dim)
        else:
            raise ValueError(
                f"Unexpected MLP head policy ndim={active_policy.ndim} at layer {self.layer_index}."
            )
        return x * gate.to(dtype=x.dtype)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        if self.impl_type == "mlp":
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = torch.nn.functional.silu(x1) * x2
        return self.w3(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_input_gate(x)
        return self._forward_impl(x)

    def forward_list(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        gated_list = [self._apply_input_gate(x) for x in x_list]
        x_flat, shapes, num_tokens = cat_keep_shapes(gated_list)
        x_flat = self._forward_impl(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class PolicyAwareSelfAttention(nn.Module):
    def __init__(self, attn: nn.Module, layer_index: int, apply_mode: str = "output_gate"):
        super().__init__()
        self.layer_index = int(layer_index)
        self.num_heads = int(attn.num_heads)
        self.scale = getattr(attn, "scale", None)
        self.qkv = attn.qkv
        self.attn_drop = getattr(attn, "attn_drop", nn.Identity())
        self.proj = attn.proj
        self.proj_drop = getattr(attn, "proj_drop", nn.Identity())
        self._runtime_head_policy = None
        if apply_mode not in ("output_gate", "attn_identity"):
            raise ValueError(
                f"head_policy_apply_mode must be 'output_gate' or 'attn_identity', "
                f"got {apply_mode!r}"
            )
        self.apply_mode = apply_mode

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

        active_policy = self._runtime_head_policy if head_policy is None else head_policy

        # Apply gate before attention computation. This matches the
        # attention-side masking used by AdaViT-style head policies, but it
        # does not physically prune the qkv/proj compute.
        if active_policy is not None:
            # Support both per-image [B, H] and per-token [B, N, H] gates.
            if active_policy.ndim == 2:
                # per-image: [B, H] → [B, H, 1, 1] for broadcast over [B, H, N, Dh]
                if active_policy.shape != (batch_size, self.num_heads):
                    raise ValueError(
                        f"Runtime head policy shape mismatch at layer {self.layer_index}: "
                        f"expected {(batch_size, self.num_heads)}, got {tuple(active_policy.shape)}."
                    )
                gate_qkv = active_policy[:, :, None, None].to(dtype=q.dtype)
            elif active_policy.ndim == 3:
                # per-token: [B, N, H] → [B, H, N, 1] for broadcast over [B, H, N, Dh]
                gate_qkv = active_policy.permute(0, 2, 1).unsqueeze(-1).to(dtype=q.dtype)
            else:
                raise ValueError(
                    f"Unexpected head policy ndim={active_policy.ndim} at layer {self.layer_index}."
                )

            if self.apply_mode == "output_gate":
                # QKV-gated approximation of AdaViT width_select on the
                # attention branch. Dropped heads have Q=K=V=0.
                q = q * gate_qkv
                k = k * gate_qkv
                v = v * gate_qkv
            else:
                # attn_identity mode: we need v BEFORE gating for the
                # identity blend after attention. Store ungated v.
                pass

        # Store ungated v for attn_identity mode
        self._last_v = v

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
        x = x.transpose(1, 2)  # [B, N, num_heads, head_dim]

        # Post-attention policy application (only for attn_identity mode)
        if active_policy is not None and self.apply_mode == "attn_identity":
            # x shape: [B, N, H, Dh]. Gate needs to broadcast accordingly.
            if active_policy.ndim == 2:
                gate_out = active_policy[:, None, :, None].to(dtype=x.dtype)  # [B, 1, H, 1]
            else:
                gate_out = active_policy[:, :, :, None].to(dtype=x.dtype)  # [B, N, H, 1]
            v_id = self._last_v.transpose(1, 2)  # [B, N, H, head_dim]
            x = gate_out * x + (1.0 - gate_out) * v_id

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
