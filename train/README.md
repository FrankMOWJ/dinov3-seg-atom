# MoS2 训练

先修改 [config.json](./config.json) 中的数据目录和 `backbone_weights`，然后在 `PDZSeg` 目录运行：

```bash
python train/train.py
```

也可以使用另一份配置，或从当前输出目录恢复训练：

```bash
python train/train.py --config train/config.json
python train/train.py --config train/config.json --resume runs/mos2/last.pth
```

配置中的相对路径按 **配置文件所在目录**解析；默认输出目录 `../runs/mos2` 对应 `PDZSeg/runs/mos2`。默认数据路径沿用旧 `Model` 的服务器路径，需要改成实际位置。关闭 Part 1 时设置 `part1_images_dirs: []`；关闭 Part 2/3 时将对应目录设为 `null`。已启用但不存在的目录会直接报错。

## 数据处理

处理逻辑来自 `Model/scripts/train.py`，独立放在本目录，不依赖旧模型代码。

| 来源 | 输入格式 | 处理 |
| --- | --- | --- |
| Part 1 | `images/*.npy`，可选 `images_old/*.npy`；对应 `masks/<stem>_mask.npy` | 读取已统一 FOV 的真实 patch |
| Part 2 | 同目录 `<stem>.png` 和 `<stem>.json` | 沿用原缩放表，点坐标画半径 8px 的圆，随机裁 256px patch |
| Part 3 | `<stem>_image.npy`；`masks/<stem>_mask.npy` | 默认前 23 张，按 1.153 缩放，256px 无重叠裁块 |

类别为 `0=背景、1=Mo、2=S、3=V1S`；JSON 中的 `V2S` 合并到 3，标签 255 不参与损失和指标。

保留原图高位深灰度值，每个 patch 使用 `(image - median) / (MAD + 1e-8)` 并截断到 `[-5,5]`。训练增强沿用翻转、90° 旋转、高斯噪声、泊松噪声、高斯模糊和亮度/对比度扰动；验证不增强。图像输出 `[1,H,W]`，PDZSeg 内部复制成 3 通道，不再额外做 ImageNet 归一化。

默认 `image_size=256`，保持旧模型约 2nm×2nm patch 的约定。设置为 512 时，将同一 256px patch 与 mask 放大到 512，不改变对应视野；mask 使用最近邻插值。沿用的缩放参数仍需要与实际图像采集尺度匹配，可以通过 `atomcount_scale_factors` 覆盖。

训练/验证按原图 frame 分组，再按数据来源分层，目标验证比例 15%。同一 frame 的 patch 始终在同一集合，已知 `MOS2-数字` 原图在 Part 1/2 间也合并分组。其他跨来源重命名的重复图像无法仅凭文件名识别。实际 patch 比例随组大小变化；独立 frame 很少的来源可能只出现在一个集合。划分记录保存为 `split.json`。

与旧实现的其他差异：RGBA 图像只平均 RGB、忽略 alpha（旧版平均全部通道）；矩形模拟图按各自高宽缩放（旧版按高度强制缩放为正方形）；缺失目录/配对标签、非有限图像、非法类别和过小的裁块输入会报错（旧版部分情况会跳过或继续处理）。默认 256px、灰度/RGB 图像和方形模拟图的正常路径已与旧版数据函数逐像素对照。固定裁剪轴不消耗随机数，与旧版保持一致。

## 模型和训练

- 使用当前 `arch/dino_seg.py`，冻结原始 DINOv3，只训练 Q/V LoRA 和解码头。
- 默认 Base + 四层特征 + SegFormer；切换 LinearHead 时设置 `decoder_type: "linear"`，`decoder_channels: null`，可用 `use_cls_token: false` 关闭 CLS。
- 默认启用预训练，需要提供匹配型号的官方权重。`pretrained: false` 仅用于流程测试或从随机初始化开始训练；不会自动加载旧模型的 Stage 0 权重。
- 损失沿用原模型：`0.5 × 加权CE + 0.5 × 前景Dice`，CE 标签平滑 0.05，类别权重 `[0.2,1,1,2]`。
- AdamW，解码头学习率 `3e-4`，LoRA 学习率 `3e-5`，5 轮 warmup 后 cosine 衰减，梯度裁剪 1.0。
- CUDA 自动选 BF16，设备不支持时选 FP16；CPU 使用 FP32。默认最多 100 轮，验证 mIoU 连续 20 轮无提升停止，`patience: 0` 关闭早停。

旧模型的三阶段解冻策略没有直接照搬，因为当前模型使用 LoRA。验证指标按整个验证集累计混淆矩阵计算，mIoU 包含背景，忽略完全未出现的类别；因此与旧脚本逐 batch 平均且空类别算 1 的数值口径不同。

## 输出

- `best.pth`：验证 mIoU 最优的 LoRA 与解码头权重，可用 `PDZSeg.load_parameters()` 加载。
- `last.pth`：同样的模型权重，加上 epoch、optimizer、scheduler、scaler、随机状态和数据划分，用于断点续训。
- `training_log.csv`：训练/验证损失、各类 IoU、mIoU、准确率、学习率和耗时。
- `config.json`、`split.json`、`summary.json`、`train.log`：本次配置、分组划分和运行结果。

两份 checkpoint 都不保存冻结的官方 backbone；推理和恢复时需要使用相同的官方预训练权重及模型配置。恢复训练沿用原 `output_dir`，检查数据划分、模型和损失配置。训练开始后修改数据、归一化或标签配置应另开输出目录。
