# Domain Self-Adaptive CTTA

这是论文《Shared & Domain Self-Adaptive Experts with Frequency-Aware
Discrimination for Continual Test-Time Adaptation》的官方代码整理版。

论文链接：

- [AAAI-26 官方论文页面](https://ojs.aaai.org/index.php/AAAI/article/view/40102)
- [arXiv:2507.00502](https://arxiv.org/abs/2507.00502)

本项目实现了共享专家、域自适应专家和频率感知域判别器，用于持续测试时
自适应（Continual Test-Time Adaptation, CTTA）。数据集和模型 checkpoint
不包含在 GitHub 代码仓库中。

## 环境

代码在以下环境中测试通过：

- Python 3.9
- CUDA 11.x
- NVIDIA A800 80 GB
- 具体 Python 依赖见 [requirements.txt](requirements.txt)

建议先安装与本机 CUDA 匹配的 PyTorch 和 torchvision，再安装其余依赖。

```bash
pip install -r requirements.txt
```

当前实验环境使用的主要版本为：

```text
torch==1.10.0
torchvision==0.11.1
timm==0.4.12
numpy==1.19.5
Pillow==8.4.0
PyYAML==6.0
tqdm==4.56.2
```

## 模型 Checkpoint

实验使用 ViT-B/16 模型，输入分辨率为 224，模型包含 12 个 Transformer
Block。

### Checkpoint 的生成方式

为了初始化我们的专家模块，我们以 ImageNet 预训练的 ViT-B/16 为基础加入
MoE，并在 ImageNet 训练集上进行若干个 epoch 的微调预热。得到的 checkpoint
同时包含完整的源模型参数和预热后的专家模块。在线测试时，共享专家以及后续
检测到新域后新增的域自适应专家，都从这个预热专家模块初始化。

当前上传到 Hugging Face 的文件为：

```text
vit_source_finetuned_imagenet_lr0.0001_freeze_True_epoch8_scalar10.0_.pt
```

[下载 MoE 预热 checkpoint](https://huggingface.co/jianchao123/Domain-Self-Adaptive-CTTA/resolve/main/vit_source_finetuned_imagenet_lr0.0001_freeze_True_epoch8_scalar10.0_.pt)

## 数据准备

`--data_root` 需要包含 ImageNet+、ImageNet++ 和 ImageNet-C 数据：

```text
tta_datasets/
├── ImageNet-C/
│   └── <corruption>/<severity>/<synset>/<image>.JPEG
├── imagenet++/
│   ├── imagenetv2-top-images-format-val/
│   └── imagenetv2-threshold0.7-format-val/
├── domain_gen/
│   ├── imagenetv2/imagenetv2-matched-frequency-format-val/
│   ├── imagenet-adversarial/imagenet-a/
│   ├── imagenet-sketch/images/
│   └── imagenet-rendition/imagenet-r/
└── imagenet_class_index.json
```

ImageNet-V2 的类别目录名为数字 `0` 到 `999`。ImageNet-C、ImageNet-A、
ImageNet-S 和 ImageNet-R 使用 ImageNet synset 目录名，例如：

```text
n01440764/
n01443537/
```

ImageNet-C 的样本顺序和标签映射文件位于项目的 `metadata/` 目录中，代码会
按照标准 ImageNet 验证集顺序读取样本，默认取每类 corruption 的前 5,000 张。

## ImageNet-C

ImageNet-C 使用 severity 5，按照原始 15 类 corruption 顺序依次处理。每类使用
5,000 张图片，batch size 为 50，每个 batch 更新 1 步。只在第一个 corruption
前 reset，后续 corruption 之间不 reset。

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 python imagenetc.py \
  --data_root /path/to/tta_datasets \
  --checkpoint /path/to/vit_source_finetuned_imagenet_lr0.0001_freeze_True_epoch8_scalar10.0_.pt \
  --save_dir /path/to/results/imagenetc
```

## ImageNet+ 和 ImageNet++

ImageNet+：

```bash
CUDA_VISIBLE_DEVICES=0 python imagenet_plus.py \
  --data_root /path/to/tta_datasets \
  --checkpoint /path/to/vit_source_finetuned_imagenet_lr0.0001_freeze_True_epoch8_scalar10.0_.pt \
  --save_dir /path/to/results/imagenet_plus
```

ImageNet++：

```bash
CUDA_VISIBLE_DEVICES=0 python imagenet_plusplus.py \
  --data_root /path/to/tta_datasets \
  --checkpoint /path/to/vit_source_finetuned_imagenet_lr0.0001_freeze_True_epoch8_scalar10.0_.pt \
  --save_dir /path/to/results/imagenet_plusplus
```

ImageNet+ 和 ImageNet++ 的默认 batch size 为 200，每个 batch 更新 1 步。
其他支持的参数可以通过重复使用 `--override KEY=VALUE` 传入。

三个入口都使用单进程 `torch.nn.DataParallel`。双卡运行示例：

```bash
CUDA_VISIBLE_DEVICES=5,6 python imagenet_plusplus.py \
  --data_root /path/to/tta_datasets \
  --checkpoint /path/to/vit_source_finetuned_imagenet_lr0.0001_freeze_True_epoch8_scalar10.0_.pt \
  --save_dir /path/to/results/imagenet_plusplus
```

也可以通过环境变量设置路径：

```bash
export IMAGENET_DATA_ROOT=/path/to/tta_datasets
export MOE_CHECKPOINT=/path/to/vit_source_finetuned_imagenet_lr0.0001_freeze_True_epoch8_scalar10.0_.pt
```

## 项目文件

- `imagenetc.py`：ImageNet-C 入口。
- `imagenet_plus.py`：ImageNet+ 入口。
- `imagenet_plusplus.py`：ImageNet++ 入口。
- `protocol_common.py`：ImageNet+ 和 ImageNet++ 的数据顺序、类别映射和在线评估逻辑。
- `moe_runtime.py`、`moe.py`、`inject_moe_1.py`、`adapter.py`：MoE 和 adapter 实现。
- `metadata/`：ImageNet-C 的样本顺序和标签映射元数据。
- `robustbench/model_zoo/`：当前 ViT 所需的模型实现。
- `cfgs/vit/moe_new.yaml`：模型和优化器基础配置。

## 许可证

本项目原创代码使用 MIT License。部分模型代码来自第三方项目，请同时遵守
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 中列出的上游许可证要求。

## 引用

```bibtex
@inproceedings{zhao2026shared,
  title={Shared and Domain Self-Adaptive Experts with Frequency-Aware Discrimination for Continual Test-Time Adaptation},
  author={Zhao, Jianchao and Ding, Chenhao and Dong, SongLin and Li, Jiangyang and Wang, Qiang and He, Yuhang and Gong, Yihong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={34},
  pages={28697--28705},
  year={2026},
  doi={10.1609/aaai.v40i34.40102}
}
```
