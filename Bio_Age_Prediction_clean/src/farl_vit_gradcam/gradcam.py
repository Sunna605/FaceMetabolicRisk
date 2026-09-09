from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def attention_gradcam(model, input_tensor, target_class=1):
    """Return per-sample 14x14 scheme-1 CAMs and logits."""
    block = model.feature_extraction.transformer.resblocks[-1]
    block.set_capture_attention(False)
    with torch.no_grad():
        reference = model(input_tensor)["cls_logits"]

    block.set_capture_attention(True)
    model.zero_grad(set_to_none=True)
    logits = model(input_tensor)["cls_logits"]
    logits[:, target_class].sum().backward()
    attention = block.get_attention_map()
    gradients = block.get_attention_gradients()
    if attention is None or gradients is None:
        raise RuntimeError("The final attention map or gradient was not captured")
    error = float((reference - logits.detach()).abs().max().item())
    if not torch.allclose(reference, logits.detach(), rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"The captured path changed logits (max error={error:.3e})")

    patch_attention = attention[:, :, 0, 1:]
    patch_gradients = gradients[:, :, 0, 1:]
    n_patches = patch_attention.shape[-1]
    grid = int(n_patches**0.5)
    if grid * grid != n_patches:
        raise RuntimeError(f"Expected a square patch grid, got {n_patches}")
    weights = patch_gradients.mean(dim=-1, keepdim=True)
    cams = (patch_attention * weights).mean(dim=1).clamp(min=0)
    cams = cams.reshape(cams.shape[0], grid, grid)
    flat = cams.flatten(1)
    shifted = cams - flat.min(dim=1).values[:, None, None]
    maxima = shifted.flatten(1).max(dim=1).values
    valid = maxima > 0
    normalized = torch.zeros_like(shifted)
    normalized[valid] = shifted[valid] / maxima[valid, None, None]
    block.set_capture_attention(False)
    return (
        normalized.detach().cpu().numpy(),
        logits.detach().cpu().numpy(),
        valid.detach().cpu().numpy(),
        error,
    )


def render_heatmap(cam, size=224):
    image = Image.fromarray(np.uint8(np.clip(cam, 0, 1) * 255)).resize(
        (size, size), resample=Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return plt.get_cmap("jet")(array)[..., :3]


def save_cam_bundle(output_dir, image, cam, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB").resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    heatmap = render_heatmap(cam)
    np.save(output_dir / "cam_14x14.npy", cam.astype(np.float32))
    image.save(output_dir / "original.png")
    plt.imsave(output_dir / "heatmap.png", heatmap)
    plt.imsave(output_dir / "overlay.png", np.clip(0.5 * image_array + 0.5 * heatmap, 0, 1))
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
