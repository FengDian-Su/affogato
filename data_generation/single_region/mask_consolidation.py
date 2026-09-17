"""Production per-view consolidation of independent per-point SAM soft masks."""
import numpy as np

DEFAULT_MASK_TAU = 0.1


def conditional_mean_masks(masks, image_shape, tau=DEFAULT_MASK_TAU):
    """Average original scores strictly above tau, independently per pixel.

    Only valid SAM requests belong in masks: absent point slots are excluded.
    The denominator is the number of masks with M(u) > tau at pixel u, NOT
    the total mask count K or the view count. No contributors gives zero.
    Always return independent float32 storage, including zero/single-mask
    cases, because subsequent cross-role processing mutates heatmaps.

    This is the conditional-mean method validated by the frozen 20260913
    consolidation pilot, promoted without depending on the ablation package.
    """
    if not np.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("tau must be finite and in [0, 1]")
    shape = tuple(image_shape)
    if len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or
                              isinstance(n, (bool, np.bool_)) or n <= 0 for n in shape):
        raise ValueError("image_shape must be a positive integer (height, width)")
    total = np.zeros(shape, dtype=np.float32)
    count = np.zeros(shape, dtype=np.int32)
    threshold = np.float32(tau)
    for mask in masks:
        x = np.asarray(mask, dtype=np.float32)
        if x.shape != shape or not np.isfinite(x).all():
            raise ValueError("masks must match image_shape and be finite")
        if (x < 0).any() or (x > 1).any():
            raise ValueError("mask scores must be in [0, 1]")
        keep = x > threshold
        total += np.where(keep, x, np.float32(0))
        count += keep
    np.divide(total, count, out=total, where=count > 0)
    return total
