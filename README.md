# PyTorch Implementation of MattNet

## Introduction

This repository is Pytorch implementation of [MattNet: Modular Attention Network for Referring Expression Comprehension](https://arxiv.org/pdf/1801.08186.pdf) in [CVPR 2018](http://cvpr2018.thecvf.com/), which achieves state-of-art performance for both bounding-box localization and segmentation tasks.

Refering Expressions are natural language utterances that indicate particular objects within a scene, e.g., "the woman in red sweater", "the man on the right", etc.
For robots or other intelligent agents communicating with people in the world, the ability to accurately comprehend such expressions will be a necessary component for natural interactions.
In this project, we address referring expression comprehension: localizing an image region described by a natural language expression. 
Examples are shown as follows:

<p align="center">
  <img src="http://bvisionweb1.cs.unc.edu/licheng/MattNet/mattnet_example.jpg" width="80%"/>
</p>

## Prerequisites

* Python 2.7
* Pytorch 0.2 or higher
* CUDA 8.0 or higher
* requirements.txt


## Installation

1. Clone the MattNet repository
```
git clone --recursive https://github.com/lichengunc/MattNet
```

2. Prepare the submodules and associated data

* Mask R-CNN: Follow the instructions of my [mask-faster-rcnn](https://github.com/lichengunc/mask-faster-rcnn) repo, preparing everything needed for `pyutils/mask-faster-rcnn`.
You could use `cv/mrcn_detection.ipynb` to test if you've get Mask R-CNN ready.

* REFER data: Use the download links of [REFER](https://github.com/lichengunc/refer) and go to the foloder running `make`. Follow `data/README.md` to prepare images and refcoco/refcoco+/refcocog annotations.

* refer-parser2: Follow the instructions of [refer-parser2](https://github.com/lichengunc/refer-parser2) to extract the parsed expressions using [Vicente's R1-R7 attributes](http://tamaraberg.com/papers/referit.pdf). **Note** this sub-module is only used if you want to train the models by yourself.


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
During training, you may want to use `cv/inpect_cv.ipynb` to check the training/validation curves and do cross validation.

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

## Pre-trained Models

In order to get the results in our paper, please follow [Training Step 1-3](#training) for data and feature preparation then run [Evaluation Step 1](#evaluation).
We provide the pre-trained models for RefCOCO, RefCOCO+ and RefCOCOg. Download and put them under `./output` folder.

1) RefCOCO: [Pre-trained model (56M)](http://bvision.cs.unc.edu/licheng/MattNet/pretrained/refcoco_unc.zip)
<table>
<tr><th> Localization (gt-box) </th><th> Localization (Mask R-CNN) </th><th> Segmentation (Mask R-CNN) </th></tr>
<tr><td>

| val | test A | test B |
|--|--|--|
| 85.57\% | 85.95\% | 84.36\% |
</td><td>

| val | test A | test B |
|--|--|--|
| 76.65\% | 81.14\% | 69.99\% |
</td><td>

| val | test A | test B |
|--|--|--|
| 75.16\% | 79.55\% | 68.87\% |
</td></tr> </table>

2) RefCOCO+: [Pre-trained model (56M)](http://bvision.cs.unc.edu/licheng/MattNet/pretrained/refcoco+_unc.zip)
<table>
<tr><th> Localization (gt-box) </th><th> Localization (Mask R-CNN) </th><th> Segmentation (Mask R-CNN) </th></tr>
<tr><td>

| val | test A | test B |
|--|--|--|
| 71.71\% | 74.28\% | 66.27\% |
</td><td>

| val | test A | test B |
|--|--|--|
| 65.33\% | 71.62\% | 56.02\% |
</td><td>

| val | test A | test B |
|--|--|--|
| 64.11\% | 70.12\% | 54.82\% |
</td></tr> </table>

3) RefCOCOg: [Pre-trained model (58M)](http://bvision.cs.unc.edu/licheng/MattNet/pretrained/refcocog_umd.zip) 
<table>
<tr><th> Localization (gt-box) </th><th> Localization (Mask R-CNN) </th><th> Segmentation (Mask R-CNN) </th></tr>
<tr><td>

| val | test |
|--|--|
| 78.96\% | 78.51\% |
</td><td>

| val | test |
|--|--|
| 66.58\% | 67.27\% |
</td><td>

| val | test |
|--|--|
| 64.48\% | 65.60\% |
</td></tr> </table>


## Demo

Run `cv/example_demo.ipynb` for demo example. 
You can also check our **[Online Demo](http://gpuvision.cs.unc.edu/refer/comprehension)**.


## Citation

    @article{yu2018mattnet,
      title={MAttNet: Modular Attention Network for Referring Expression Comprehension},
      author={Yu, Licheng and Lin, Zhe and Shen, Xiaohui and Yang, Jimei and Lu, Xin and Bansal, Mohit and Berg, Tamara L},
      journal={arXiv preprint arXiv:1801.08186},
      year={2018}
    }

## License

MattNet is released under the MIT License (refer to the LICENSE file for details).

## Authorship

This project is maintained by [Licheng Yu](http://cs.unc.edu/~licheng/).


