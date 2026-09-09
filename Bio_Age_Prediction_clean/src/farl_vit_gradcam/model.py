from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    """FaRL-compatible layer normalization with float32 statistics."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * x.to(dtype) + self.bias


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = LayerNorm(width)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(width, width * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(width * 4, width)),
                ]
            )
        )
        self.ln_2 = LayerNorm(width)
        self.capture_attention = False
        self.attention_map = None
        self.attention_gradients = None

    def set_capture_attention(self, enabled=True):
        self.capture_attention = enabled
        self.attention_map = None
        self.attention_gradients = None

    def get_attention_map(self):
        return self.attention_map

    def get_attention_gradients(self):
        return self.attention_gradients

    def _save_attention_gradients(self, gradients):
        self.attention_gradients = gradients

    def _attention_with_maps(self, x):
        sequence_length, batch_size, embed_dim = x.shape
        heads = self.attn.num_heads
        head_dim = embed_dim // heads
        qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.contiguous().view(
                sequence_length, batch_size, heads, head_dim
            ).permute(1, 2, 0, 3)

        query = split_heads(query) * (head_dim**-0.5)
        key = split_heads(key)
        value = split_heads(value)
        attention = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)), dim=-1
        )
        self.attention_map = attention
        if attention.requires_grad:
            attention.register_hook(self._save_attention_gradients)
        output = torch.matmul(attention, value)
        output = output.permute(2, 0, 1, 3).contiguous().view(
            sequence_length, batch_size, embed_dim
        )
        return F.linear(output, self.attn.out_proj.weight, self.attn.out_proj.bias)

    def attention(self, x):
        if self.capture_attention:
            return self._attention_with_maps(x)
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution=224, patch_size=16, width=768, layers=12, heads=12, output_dim=512):
        super().__init__()
        self.num_features = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        sequence_length = (input_resolution // patch_size) ** 2 + 1
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(sequence_length, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = self.ln_pre(torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype))
        x = x.permute(1, 0, 2)
        for block in self.transformer.resblocks:
            x = block(x)
        x = self.ln_post(x.permute(1, 0, 2)[:, 0, :])
        return x @ self.proj


def _checkpoint_state(path):
    payload = torch.load(Path(path), map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    has_visual_prefix = any(
        key.startswith("visual.") or key.startswith("module.visual.")
        for key in state
    )
    normalized = {}
    for key, value in state.items():
        if has_visual_prefix and key.startswith("module.visual."):
            key = key[len("module.visual.") :]
        elif has_visual_prefix and key.startswith("visual."):
            key = key[len("visual.") :]
        elif has_visual_prefix:
            continue
        elif key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    return normalized


def load_farl(checkpoint_path, device="cpu"):
    model = VisualTransformer()
    state = _checkpoint_state(checkpoint_path)
    expected = model.state_dict()
    state = {key: value for key, value in state.items() if key in expected}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"FaRL checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device)


class FaRLClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.feature_extraction = encoder
        features = encoder.num_features
        self.classification_head = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(inplace=True),
            nn.Linear(features, num_classes),
        )

    def forward(self, x, return_features=False):
        features = self.feature_extraction(x)
        logits = self.classification_head(features)
        result = {"cls_logits": logits}
        return (result, features) if return_features else result

    def load_checkpoint(self, checkpoint_path):
        payload = torch.load(Path(checkpoint_path), map_location="cpu")
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        normalized = {}
        for key, value in state.items():
            while key.startswith("module."):
                key = key[len("module.") :]
            normalized[key] = value
        missing, unexpected = self.load_state_dict(normalized, strict=False)
        if missing:
            raise RuntimeError(f"Downstream checkpoint is missing keys: {missing}")
        return unexpected


def load_farl_classifier(farl_checkpoint, model_checkpoint=None, device="cpu"):
    model = FaRLClassifier(load_farl(farl_checkpoint, device=device))
    if model_checkpoint is not None:
        model.load_checkpoint(model_checkpoint)
    return model.to(device)
