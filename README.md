
# Clean ImageNet MoE Project

This directory is the minimal runnable copy for the paper method:
"Shared & Domain Self-Adaptive Experts with Frequency-Aware Discrimination
for Continual Test-Time Adaptation".

The original project is left unchanged. Datasets and model checkpoints are
intentionally kept outside this repository.

Paper: [AAAI-26 proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/40102)
and [arXiv:2507.00502](https://arxiv.org/abs/2507.00502).

## Environment

The code was tested with Python 3.9, CUDA 11.x, and the packages listed in
`requirements.txt`. Install the matching PyTorch and torchvision builds for
the CUDA version on your machine.

## Entry points

ImageNet-C:

```bash
CUDA_VISIBLE_DEVICES=0 python imagenetc.py \
  --data_root /path/to/tta_datasets \
  --checkpoint /path/to/source_checkpoint.pt \
  --save_dir /path/to/results/imagenetc
```

The ImageNet-C entry uses severity 5, the original 15-corruption order, 5,000
images per corruption, batch size 50, and one update step per batch. It resets
only before the first corruption and keeps the adapted model across the
remaining corruptions. Its default `best` preset is the reported ImageNet-C
configuration; use `--config original` for the original settings.

ImageNet+:

```bash
CUDA_VISIBLE_DEVICES=0 python imagenet_plus.py \
  --data_root /path/to/tta_datasets \
  --checkpoint /path/to/source_checkpoint.pt \
  --save_dir /path/to/results/imagenet_plus
```

ImageNet++:

```bash
CUDA_VISIBLE_DEVICES=0 python imagenet_plusplus.py \
  --data_root /path/to/tta_datasets \
  --checkpoint /path/to/source_checkpoint.pt \
  --save_dir /path/to/results/imagenet_plusplus
```

Both commands default to the current best configuration. Use
`--config original` for the original settings. Any supported nonzero
parameter can be overridden with repeated `--override KEY=VALUE` options.
Batch size is fixed at 200 and update steps are fixed at 1.

The two paths can also be supplied through environment variables:
`IMAGENET_DATA_ROOT` and `MOE_CHECKPOINT`.

For two GPUs, expose only the selected cards, for example:

```bash
CUDA_VISIBLE_DEVICES=5,6 python imagenet_plusplus.py \
  --data_root /path/to/tta_datasets \
  --checkpoint /path/to/source_checkpoint.pt \
  --save_dir /path/to/results/imagenet_plusplus
```

The entry points use single-process `torch.nn.DataParallel`.

## Data Layout

`--data_root` must contain these ImageFolder datasets and the class mapping:

```text
tta_datasets/
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

ImageNet-V2 class directories are numeric (`0` through `999`). ImageNet-A,
ImageNet-S, and ImageNet-R use ImageNet synset directory names. The images
are resized to 256, center-cropped to 224, and converted to tensors.

The source checkpoint is not included. It must be supplied separately and
should be the ViT-B/16 source checkpoint used by the experiments. The
checkpoint will be hosted in the companion Hugging Face repository:
`https://huggingface.co/ZJC25127/Domain-Self-Adaptive-CTTA`.

For ImageNet-C, also place the dataset at:

```text
tta_datasets/ImageNet-C/<corruption>/<severity>/<synset>/<image>.JPEG
```

The ImageNet-C metadata files shipped in `metadata/` preserve the standard
50,000-image ImageNet validation order. The runner selects the first 5,000
entries for each corruption by default.

## Included files

- `imagenetc.py`, `imagenet_plus.py`, `imagenet_plusplus.py`: the three executable entries.
- `protocol_common.py`: dataset ordering, class mapping, online metric, and reset protocol.
- `metadata/`: small ImageNet-C sample-order and label-mapping files.
- `moe_runtime.py`, `moe.py`, `inject_moe_1.py`, `adapter.py`: the MoE implementation.
- `robustbench/model_zoo/`: only the ViT model implementation needed by this project.
- `cfgs/vit/moe_new.yaml`: model and optimizer base settings.

The copy intentionally excludes TENT, CoTTA, ViDA, pretraining scripts,
visualization scripts, historical logs, analysis artifacts, and unrelated
dataset code.

## License

Original project code is released under the MIT License. Some model code is
adapted from third-party projects; see `THIRD_PARTY_NOTICES.md` and retain
those projects' respective license terms.

## Citation

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
