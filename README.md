# PatchAlign3D: Local Feature Alignment for Dense 3D Shape Understanding

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data](#data)
- [Inference](#inference)
- [Training](#training)
- [Citation](#citation)

## Introduction
Official source code for PatchAlign3D: Local Feature Alignment for Dense 3D Shape Understanding


## Installation
Install your CUDA-enabled PyTorch separately. You also need `pointnet2_ops` and `knn_cuda`.

```
conda create -n patchalign3d python=3.9
conda activate patchalign3d
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

```
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Data
Details about data download can be found in [here](src/datasets/README.md)

## Inference
Run inference on a single shape and save per-point predictions.

```
python patchalign3d/inference/infer.py \
  --ckpt /path/to/stage2_last.pt \
  --input /path/to/shape.npz \
  --labels "seat,back,leg,arm"
```

### Checkpoints
The stage-2 (PatchAlign3D Model) checkpoint is available on [Hugging Face](https://huggingface.co/patchalign3d/patchalign3d-encoder).

### Evaluation (ShapeNetPart)
```
python patchalign3d/inference/eval.py \
  --ckpt /path/to/stage2_last.pt \
  --shapenet_root /path/to/shapenetcore_partanno_segmentation_benchmark_v0_normal \
  --gpu 0 --num_group 128 --group_size 32 \
  --clip_model ViT-bigG-14 --clip_pretrained laion2b_s39b_b160k
```

## Training

### Stage 0 (optional): Offline DINO patch features
If you want to regenerate DINO features, clone COPS into `PatchAlign3D/cops` and run:

```
python patchalign3d/tools/precompute_dino_patch_features.py \
  --root /path/to/data_root \
  --out_dir_name patch_dino \
  --num_views 10 --view_batch 2 
```

### Offline text bank (optional but recommended)
```
python patchalign3d/tools/build_text_bank.py \
  --data_root /path/to/data_root \
  --train_list /path/to/train.txt \
  --val_list /path/to/val.txt \
  --mode both \
  --clip_model ViT-B-16 --clip_pretrained laion2b_s34b_b88k \
  --batch_texts 256 
```

### Stage 1 (visual alignment)
```
python patchalign3d/training/stage1.py \
  --data_root /path/to/data_root \
  --train_list /path/to/train.txt \
  --val_list /path/to/val.txt \
  --gpu 0 --batch_size 32 --epoch 100 \
  --eval_every 2 --save_every 10 \
  --npoint 2048 --num_group 128 --group_size 32 \
  --random_sample_train --train_encoder \
  --dino_feature_subdir patch_dino \
  --wandb_project part_segmentation \
  --wandb_run_name stage1_dino --wandb_mode online
```

### Stage 2 (text alignment)
```
python patchalign3d/training/stage2.py \
  --data_root /path/to/data_root \
  --train_list /path/to/train.txt \
  --val_list /path/to/val.txt \
  --batch_size 32 --epoch 100 \
  --eval_every 2 --save_every 10 \
  --gpu 0 --clip_tau 0.07 \
  --exclude_category_label --random_sample_train \
  --npoint 2048 --drop_labels_not_in_bank --text_bank_require \
  --num_group 128 --group_size 32 \
  --init_stage1 /path/to/stage1_last.pt \
  --train_last_block_only \
  --wandb_project part_segmentation \
  --wandb_run_name stage2_clip_patch_bce
```

## Citation
```
@article{patchalign3d2025,
  title={PatchAlign3D: Local Feature Alignment for Dense 3D Shape Understanding},
  author={},
  journal={},
  year={}
}
```
