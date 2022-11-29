# PyTorch Implementation of MAttNet

## Introduction

This repository is Pytorch implementation of [MAttNet: Modular Attention Network for Referring Expression Comprehension](https://arxiv.org/pdf/1801.08186.pdf) in [CVPR 2018](http://cvpr2018.thecvf.com/).
Refering Expressions are natural language utterances that indicate particular objects within a scene, e.g., "the woman in red sweater", "the man on the right", etc.
For robots or other intelligent agents communicating with people in the world, the ability to accurately comprehend such expressions will be a necessary component for natural interactions.
In this project, we address referring expression comprehension: localizing an image region described by a natural language expression. 
Check our [paper](https://arxiv.org/pdf/1801.08186.pdf) and [online demo](http://vision2.cs.unc.edu/refer/comprehension) for more details.
Examples are shown as follows:

<p align="center">
  <img src="http://bvisionweb1.cs.unc.edu/licheng/MattNet/mattnet_example.jpg" width="75%"/>
</p>

## Prerequisites

* Python 2.7
* Pytorch 0.2 (may not work with 1.0 or higher)
* CUDA 8.0

## Installation

1. Clone the MAttNet repository

```
git clone --recursive https://github.com/lichengunc/MAttNet
```

2. Prepare the submodules and associated data

* Mask R-CNN: Follow the instructions of my [mask-faster-rcnn](https://github.com/lichengunc/mask-faster-rcnn) repo, preparing everything needed for `pyutils/mask-faster-rcnn`.
You could use `cv/mrcn_detection.ipynb` to test if you've get Mask R-CNN ready.

* REFER API and data: Use the download links of [REFER](https://github.com/lichengunc/refer) and go to the foloder running `make`. Follow `data/README.md` to prepare images and refcoco/refcoco+/refcocog annotations.

* refer-parser2: Follow the instructions of [refer-parser2](https://github.com/lichengunc/refer-parser2) to extract the parsed expressions using [Vicente's R1-R7 attributes](http://tamaraberg.com/papers/referit.pdf). **Note** this sub-module is only used if you want to train the models by yourself.


## Training
1. Prepare the training and evaluation data by running `tools/prepro.py`:

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
CUDA_VISIBLE_DEVICES=gpu_id python tools/run_detect.py --dataset refcoco --splitBy unc --conf_thresh 0.65
CUDA_VISIBLE_DEVICES=gpu_id python tools/run_detect_to_mask.py --dataset refcoco --splitBy unc
CUDA_VISIBLE_DEVICES=gpu_id python tools/extract_mrcn_det_feats.py --dataset refcoco --splitBy unc
```

4. Train MAttNet with ground-truth annotation:

```bash
./experiments/scripts/train_mattnet.sh GPU_ID refcoco unc
```
During training, you may want to use `cv/inpect_cv.ipynb` to check the training/validation curves and do cross validation.

## Evaluation

Evaluate MAttNet with ground-truth annotation:

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

1) RefCOCO: [Pre-trained model (56M)](http://bvisionweb1.cs.unc.edu/licheng/MattNet/pretrained/refcoco_unc.zip)
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

2) RefCOCO+: [Pre-trained model (56M)](http://bvisionweb1.cs.unc.edu/licheng/MattNet/pretrained/refcoco+_unc.zip)
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

3) RefCOCOg: [Pre-trained model (58M)](http://bvisionweb1.cs.unc.edu/licheng/MattNet/pretrained/refcocog_umd.zip) 
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


## Pre-computed detections/masks
We provide the [detected boxes/masks](http://bvisionweb1.cs.unc.edu/licheng/MattNet/detections.zip) for those who are interested in automatic comprehension.
This was done using [Training Step 3](#training).
**Note** our Mask R-CNN is trained on COCO’s training images, **excluding** those in RefCOCO, RefCOCO+, and RefCOCOg’s validation+testing. 
That said it is unfair to use the other off-the-shelf detectors trained on whole COCO set for this task.

## Demo

Run `cv/example_demo.ipynb` for demo example. 
You can also check our [Online Demo](http://vision2.cs.unc.edu/refer/comprehension).


## Citation

    @inproceedings{yu2018mattnet,
      title={MAttNet: Modular Attention Network for Referring Expression Comprehension},
      author={Yu, Licheng and Lin, Zhe and Shen, Xiaohui and Yang, Jimei and Lu, Xin and Bansal, Mohit and Berg, Tamara L},
      booktitle={CVPR},
      year={2018}
    }

## License

MAttNet is released under the MIT License (refer to the LICENSE file for details).


## A few notes

I'd like to share several thoughts after working on Referring Expressions for 3 years (since 2015):

* **Model Improvement**: I'm satisfied with this model architecture but still feel the context information is not fully exploited. We tried the context of visual comparison in our [ECCV2016](https://arxiv.org/pdf/1608.00272.pdf). It worked well but relied too much on the detector. That's why I removed the appearance difference in this paper. (Location comparison still remains as it's too important.) I'm looking forward to seeing more robust and interesting context proposed in the future. 
Another direction is the end-to-end multi-task training. Current model loses some concepts after going through Mask R-CNN. For example, Mask R-CNN can perfectly detect (big) ``sports ball`` in an image but MAttNet can no longer recognize it. The reason is we are training the two models seperately and our RefCOCO dataset do not have ball-related expressions.

* **Borrowing External Concepts**: Current datasets (RefCOCO, RefCOCO+, RefCOCOg) have bias toward ``person`` category. Around half of the expressions are related to person. However, in real life people may also be interested in referring other common objects (cup, bottle, book) or even stuff (sky, tree or building). As RefCOCO already provides common referring expression structure, the (only) piece left is getting the universal objects/stuff concepts, which could be borrowed from external datasets/tasks.

* **Referring Expression Generation (REG)**: Surprisingly few paper works on referring expression generation task so far! Dialogue is important. Referring to things is always the first step for computer-to-human interaction.
(I don't think people would love to use a passive computer or robot which cannot talk.)
In our [CVPR2017](https://arxiv.org/pdf/1612.09542.pdf), we actually collected more testing expressions for better REG evaluation. (Check [REFER2](https://github.com/lichengunc/refer2) for the data. The only difference with [REFER](https://github.com/lichengunc/refer) is it contains more testing expressions on RefCOCO and RefCOCO+.)
While we achieved the SOA results in the paper, there should be plentiful space for further improvement. 
Our speaker model can only utter "boring" and "safe" expressions, thus cannot well specify every object in an image.
GAN or a Modular Speaker might be effective weapons as future work.

* **Data Collection**: Larger Referring Expressions dataset is apparently the most straight-forward way to improve the performance of any model. You might have two questions: 1) What data should we collect? 2) How do we collect the dataset? 
A larger Referring Expression dataset covering the whole MS COCO is expected (of course). 
This will also make end-to-end learning possible in the future.
Task-specific dataset is also interesting.
Since [ReferIt Game](http://tamaraberg.com/referitgame/), there have been several datasets in different domains, e.g., [video](https://arxiv.org/pdf/1708.01641v1.pdf), [dialogue](https://arxiv.org/pdf/1611.08481v2.pdf) and [spoken language](https://arxiv.org/pdf/1711.03800v2.pdf).
Note you may be careful about the problem setting. 
Randomly fitting referring expressions into a task (just for paper publication) is boring.
As for the collection method, I prefer the way used in our ealy work [ReferIt Game](http://tamaraberg.com/referitgame/). The collected expressions might be slightly short (compared with image captioning datasets), but that is how we refer things naturally in daily life.

## Authorship

This project is maintained by [Licheng Yu](http://cs.unc.edu/~licheng/).
