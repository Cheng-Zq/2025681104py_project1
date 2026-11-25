"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import json
from pathlib import Path

import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


CONFIG_PATH = Path(__file__).with_name('config.json')


def load_config(path=None):
    """Read the checkpoint configuration; geometry comes from this file.

    Returns None when the file is absent so callers can fall back to the
    shapes stored in the weight archive.
    """
    try:
        with open(Path(path) if path is not None else CONFIG_PATH, encoding='utf-8') as handle:
            return json.load(handle)
    except OSError:
        return None


def count_layers(weights):
    """Infer the number of visual Transformer blocks from the weight names."""
    prefix = 'visual.transformer.resblocks.'
    indices = {int(key[len(prefix):].split('.')[0]) for key in weights
               if key.startswith(prefix)}
    return max(indices) + 1 if indices else 12


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        batch, n_tokens, width = x.shape
        if width % self.heads:
            raise ValueError(f'width {width} is not divisible by {self.heads} heads')
        return x.reshape(batch, n_tokens, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        batch_size, n_tokens, _ = x.shape
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(np.float32(self.head_dim))

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].
        context = coefficients @ v
        merged = context.transpose(0, 2, 1, 3).reshape(batch_size, n_tokens, self.width)

        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        hidden = u @ self.fc1_weight.T + self.fc1_bias
        activated = self.gelu(hidden)
        mlp_update = activated @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights, config=None):
        config = load_config() if config is None else config
        if config is None:
            config = {}
        patch_weight = weights['visual.conv1.weight']  # [D,3,p,p]
        self.patch_size = int(config.get('patch_size', patch_weight.shape[-1]))
        self.channels = int(config.get('channels', patch_weight.shape[1]))
        width = patch_weight.shape[0]
        heads = int(config.get('heads', 12))

        self.patch_weight = patch_weight.reshape(width, -1)  # [D,3*p*p]
        self.cls = weights['visual.class_embedding'].reshape(1, 1, width)
        self.position = weights['visual.positional_embedding']  # [tokens,width]
        grid = int(round((self.position.shape[0] - 1) ** 0.5))
        self.image_size = int(config.get('image_size', grid * self.patch_size))
        expected_tokens = (self.image_size // self.patch_size) ** 2 + 1
        if self.position.shape[0] != expected_tokens:
            raise ValueError(f'positional embedding holds {self.position.shape[0]} tokens, '
                             f'expected {expected_tokens} for {self.image_size}px input')
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        layers = int(config['layers']) if 'layers' in config else count_layers(weights)
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), heads)
                       for i in range(layers)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        batch, channels, height, width = images.shape
        patch = self.patch_size
        if channels != self.channels:
            raise ValueError(f'expected {self.channels} channels, got {channels}')
        if height % patch or width % patch:
            raise ValueError(f'image size {height}x{width} is not divisible by patch {patch}')
        tokens = (height // patch) * (width // patch)
        if tokens + 1 != self.position.shape[0]:
            raise ValueError(f'{tokens} patches do not match the {self.position.shape[0] - 1} '
                             'positional-embedding patch slots')
        # Split H and W into patch grid: [B,3,H/p,p,W/p,p]
        x = images.reshape(batch, channels, height // patch, patch, width // patch, patch)
        # Bring grid axes together: [B,H/p,W/p,3,p,p]  (row-major, left to right)
        x = x.transpose(0, 2, 4, 1, 3, 5)
        # Flatten patch: [B,tokens,3*p*p]
        return x.reshape(batch, tokens, channels * patch * patch)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        patch_tokens = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image (broadcast, copied by concatenate).
        cls = np.broadcast_to(self.cls, (images.shape[0], 1, self.cls.shape[-1]))

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        tokens = np.concatenate([cls, patch_tokens], axis=1)
        return tokens + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        cls = tokens[:, 0, :]
        output = self.post_norm(cls) @ self.projection
        return unit_vector(output)
