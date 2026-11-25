# 基于 NumPy 的 OpenCLIP ViT-B-32 图像编码器

用纯 NumPy 复现 OpenCLIP ViT-B-32 图像塔的前向计算：输入 `[B,3,224,224]` 的预处理图像批次，输出 `[B,512]` 的 L2 归一化图像 token。仓库包含补全后的编码器实现、分阶段数值校验脚本，以及课程提供的权重与图片资源说明。

## 快速开始

```bash
python -m pip install -r requirements.txt
python check.py --stage all   # 8 个阶段全部 PASS 即实现正确
python run.py                 # 生成 output_tokens.npz，形状 (10, 512)
```

权重 `weights/image_encoder.npz`（约 311 MB）随课程附件 `project1-学生版.zip` 提供，因体积过大不纳入版本库（见 `.gitignore`）；10 张示例图与独立参考输出体积较小，已随仓库提供，`check.py` 可直接运行。

## 环境（实测）

| 项目 | 版本 / 说明 |
|------|-------------|
| Python | 3.13.7 |
| NumPy | 2.5.4 |
| Pillow | 13.0.0 |
| 平台 | Windows 11，CPU（不依赖 GPU / PyTorch / OpenCLIP） |
| `check.py --stage all` | 约 6.8 s |
| `run.py`（10 张图） | 约 5.7 s |

`requirements.txt` 给出可运行的版本区间，上表为实际验证过的组合。

## 校验结果

`python check.py --stage all` 的 8 个阶段全部通过，最大绝对误差远低于各自容差：

| 阶段 | 形状 | max_abs | 容差 |
|------|------|---------|------|
| assets | 152 个张量 | — | — |
| preprocess | (10,3,224,224) | 0.000e+00 | atol=2e-06 |
| patches | (2,49,3072) | 0.000e+00 | atol=2e-04 |
| tokens | (2,50,768) | 4.578e-05 | atol=2e-04 |
| block0 | (2,50,768) | 6.109e-06 | atol=2e-04 |
| last_tokens | (2,50,768) | 4.101e-05 | atol=2e-04 |
| output | (10,512) | 5.960e-07 | atol=2e-05 |
| batch | (3,512) | 4.657e-07 | atol=2e-05 |

最终输出与独立 FP32 参考的最大绝对误差为 5.96e-07，十张图余弦相似度均不低于 0.9999999。

## 代码结构

```
image_encoder.py      学生实现：patchify、split_heads、多头注意力、Transformer 块、embed、最终输出
helpers.py            提供的 LayerNorm / GELU(erf) / softmax / unit_vector
preprocess_image.py   提供的缩放、中心裁剪与归一化
runtime.py            权重与图片加载
check.py              对照 assets/reference_outputs.npz 的分阶段数值校验
run.py                命令行入口，保存输出 token
transformer_steps.py  讲义用微型 Transformer 示例
config.json           模型几何与归一化参数（patch 大小、层数、头数等）
assets/               参考输出与 10 张示例图及其校验信息
weights/              图像编码器权重（不入库，从课程附件获取）
实验报告.md / .docx    实验报告
```

实现只修改 `image_encoder.py`，其余文件保持课程提供的原样。

## 计算约定

- 补丁按行优先排列：先从左到右，再从上到下；每个 32×32 RGB 补丁内部按 C、H、W 展平为 3072 维。
- QKV、MLP、输出投影的权重按 `[out,in]` 存储，计算时使用 `weight.T`；`visual.proj` 已为 `[in,out]`，不再转置。
- 注意力得分缩放因子为 `1/sqrt(head_dim)`；softmax 使用减最大值的数值稳定形式。
- 采用预归一化结构：每个子层先 LayerNorm，残差加回子层输入（而非归一化后的结果）。
- LayerNorm epsilon 为 1e-5，激活为精确 erf 形式的 GELU，最终输出做单位化。

## 仓库地址

待补充：上传后填写 GitHub 地址（提交前请确认助教与老师有访问权限）。