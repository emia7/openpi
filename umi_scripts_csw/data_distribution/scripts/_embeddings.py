"""
Frozen SigLIP (PaliGemma image tower) helpers aligned with training / PLAN_IMAGE_ACTION_MI scheme A.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import openpi.models.model as _model


def siglip_image_module(model: nnx.Module) -> nnx.Module:
    pg = model.PaliGemma
    return pg.img if hasattr(pg, "img") else pg["img"]


def embed_image_tokens_mean(model: nnx.Module, img_bhwc: jnp.ndarray) -> jnp.ndarray:
    """Mean-pool SigLIP patch tokens -> (B, D)."""
    tokens, _ = siglip_image_module(model)(img_bhwc, train=False)
    return jnp.mean(tokens, axis=1)


def l2_normalize_np(z: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(z, axis=axis, keepdims=True)
    return z / np.maximum(n, eps)


def encode_lr_base_l2_normalized(
    model: nnx.Module, obs: _model.Observation
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Three independent forwards (left / right / base), then L2-normalize each row.
    Returns numpy (B, D) float64.
    """
    imgs = obs.images
    z_l = embed_image_tokens_mean(model, imgs["left_wrist_0_rgb"])
    z_r = embed_image_tokens_mean(model, imgs["right_wrist_0_rgb"])
    z_b = embed_image_tokens_mean(model, imgs["base_0_rgb"])
    z_ln = l2_normalize_np(np.asarray(z_l, dtype=np.float64))
    z_rn = l2_normalize_np(np.asarray(z_r, dtype=np.float64))
    z_bn = l2_normalize_np(np.asarray(z_b, dtype=np.float64))
    return z_ln, z_rn, z_bn


def load_pi_model_from_train_config(train_cfg):
    """Load Pi model with checkpoint weights (same pattern as image_action_dependence)."""
    init = train_cfg.model.create(jax.random.key(0))
    _graphdef, state = nnx.split(init)
    merged = train_cfg.weight_loader.load(state.to_pure_dict())
    return train_cfg.model.load(merged)
