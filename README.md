# MattNet
Modular Attention Network for Referring Expression Comprehension

## Introduction

This repository is Pytorch implementation of [MattNet](https://arxiv.org/pdf/1801.08186.pdf), which achieves state-of-art performance in CVPR2018 (2018.07).

## Prerequisites

* Python 2.7
* Pytorch 0.2 or higher
* CUDA 8.0 or higher
* requirements.txt


## Prepare submodules and data
* **Mask R-CNN**: Follow the instructions of my [mask-faster-rcnn](https://github.com/lichengunc/mask-faster-rcnn) repo, preparing everything needed for `pyutils/mask-faster-rcnn`.

* **REFER data**: Use the download links of [REFER](https://github.com/lichengunc/refer), preparing the images and refcoco/refcoco+/refcocog annotations under `data/`.

* **refer-parser2**: Follow the instructions of [refer-parser2](https://github.com/lichengunc/refer-parser2) to extract the parsed expressions using [Vicente's R1-R7 attributes](http://tamaraberg.com/papers/referit.pdf). **Note** this sub-module is only used if you want to reproduce the training process.


## Training
1. Prepare the training data by running `tools/prepro.py`:
```
python tools/prepro.py --dataset refcoco --splitBy unc
```

2. Extract features using Mask R-CNN, where the `head_feats` are used in subject module training and `ann_feats` is used in relationship module training.
```bash
CUDA_VISIBLE_DEVICES=gpu_id python tools/extract_mrcn_head_feats.py --dataset refcoco --splitBy unc
CUDA_VISIBLE_DEVICES=gpu_id python tools/extract_mrcn_ann_feats.py --dataset refcoco --splitBy unc
```

3. Detect objects/masks and extract features (only needed if you want to evaluate the automatic comprehension). We empirically set the confidence threshold of Mask R-CNN as 0.65.
```bash
CUDA_VISIBLE_DEVICES=id python tools/run_detect.py --dataset refcoco --splitBy unc --conf_thresh 0.65
CUDA_VISIBLE_DEVICES=id python tools/run_detect_to_mask.py --dataset refcoco --splitBy unc
CUDA_VISIBLE_DEVICES=id python tools/extract_mrcn_det_feats.py --dataset refcoco --splitBy unc
```

4. Train MattNet with ground-truth annotation:
```bash
./experiments/scripts/train_mattnet.sh GPU_ID refcoco unc
```

## Evaluation

Evaluate MattNet with ground-truth annotation:
```bash
./experiments/scripts/eval_easy.sh GPUID refcoco unc
```

If you detected/extracted the Mask R-CNN results already (step 3 above), now you can evaluate the automatic comprehension accuracy using Mask R-CNN detection and segmentation:
```bash
./experiments/scripts/eval_dets.sh GPU_ID refcoco unc
./experiments/scripts/eval_masks.sh GPU_ID refcoco unc
```









