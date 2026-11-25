"""Regenerate fig_error_comparison.png from the implementation and the reference.

The stage numbers are recomputed exactly as check.py does, so the figure in the
repository can be reproduced with:  python make_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from runtime import ROOT, load_weights, load_images
from image_encoder import ImageEncoder

STAGES = ('preprocess', 'patches', 'tokens', 'block0', 'last_tokens', 'output', 'batch')
TOL = {'preprocess': 2e-6, 'patches': 2e-4, 'tokens': 2e-4, 'block0': 2e-4,
       'last_tokens': 2e-4, 'output': 2e-5, 'batch': 2e-5}
BAR = '#4472C4'
TICK = '#C00000'


def stage_results(encoder, images, ref):
    def compute(stage):
        if stage == 'preprocess':
            return images, ref['preprocessed']
        if stage == 'patches':
            return encoder.patchify(images[:2]), ref['patches'][:2]
        if stage == 'tokens':
            return encoder.embed(images[:2]), ref['input_tokens'][:2]
        if stage == 'block0':
            return encoder.blocks[0](encoder.pre_norm(encoder.embed(images[:2]))), ref['block0_tokens'][:2]
        if stage == 'last_tokens':
            return encoder.encode_tokens(images[:2]), ref['last_tokens'][:2]
        if stage == 'output':
            return encoder(images), ref['output_tokens']
        order = [1, 0, 1]
        return encoder(images[order]), ref['output_tokens'][order]

    rows = []
    for stage in STAGES:
        actual, expected = compute(stage)
        assert actual.shape == expected.shape and actual.dtype == np.float32
        assert np.allclose(actual, expected, rtol=2e-4, atol=TOL[stage]), stage
        rows.append((stage, float(np.abs(actual - expected).max())))
    per_image = np.abs(encoder(images) - ref['output_tokens']).max(axis=1)
    return rows, per_image


def draw(rows):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(6.9, 3.0), dpi=300)
    names = [name for name, _ in rows]
    values = np.array([value for _, value in rows])
    y = np.arange(len(rows))[::-1]

    floor = 1e-7
    ceiling = 6e-4
    plotted = np.where(values > 0, values, floor)
    ax.barh(y, plotted, height=0.5, color=BAR, edgecolor='white', linewidth=0.8,
            label='最大绝对误差 max_abs')

    decades = np.log10(ceiling / floor)
    for yi, (name, value) in zip(y, rows):
        ax.vlines(TOL[name], yi - 0.26, yi + 0.26, color=TICK, linewidth=2.2,
                  label='容差阈值' if name == names[0] else None)
        if value == 0:
            ax.text(1.6 * floor, yi, '0', va='center', ha='left',
                    fontsize=10, color='#555555')
            continue
        inside = np.log10(ceiling / value) < 1.6
        if inside:
            ax.text(value * 0.55, yi, f'{value:.2e}', va='center', ha='right',
                    fontsize=10, color='white')
        else:
            ax.text(value * 1.6, yi, f'{value:.2e}', va='center', ha='left',
                    fontsize=10, color='#333333')

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11.5)
    ax.set_xscale('log')
    ax.set_xlim(floor, ceiling)
    ax.set_xlabel('误差（对数坐标）', fontsize=11)
    ax.grid(axis='x', which='major', color='#DDDDDD', linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.005),
               ncol=2, fontsize=10, frameon=False, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = ROOT / 'fig_error_comparison.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print('saved', out)


def main():
    encoder = ImageEncoder(load_weights())
    images = load_images()
    with np.load(ROOT / 'assets/reference_outputs.npz', allow_pickle=False) as ref:
        rows, per_image = stage_results(encoder, images, ref)
    for name, value in rows:
        print(f'{name:12} max_abs={value:.6e}  atol={TOL[name]:g}')
    print(f'per-image output max_abs: {per_image.min():.3e} .. {per_image.max():.3e}')
    draw(rows)


if __name__ == '__main__':
    main()