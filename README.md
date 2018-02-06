# Consensus-based Sequence Training for Video Captioning #

Code for the video captioning methods from ["Consensus-based Sequence Training for Video Captioning" (Phan, Henter, Miyao, Satoh. 2017)](https://arxiv.org/abs/1712.09532).

## Dependencies ###

* Python 2.7
* Pytorch 0.2
* [Microsoft COCO Caption Evaluation](https://github.com/tylin/coco-caption)
* [CIDEr](https://github.com/plsang/cider)

(Check out the `coco-caption` and `cider` projects into your working directory)

## Data

Data can be downloaded [here](https://drive.google.com/drive/folders/1t65uYsDck6VV045GIaJXPIqL86vSGtyQ?usp=sharing) (643 MB). This folder contains: 
* input/msrvtt: annotatated captions (note that `val_videodatainfo.json` is a symbolic link to `train_videodatainfo.json`)
* output/feature: extracted features
* output/model/cst_best: model file and generated captions on test videos of our best run (CIDEr 54.2) 

## Getting started ###

Extract video features
  - Extracted features of ResNet, C3D, MFCC and Category embeddings are shared in the above link

Generate metadata

```bash
make pre_process
```

Pre-compute document frequency for CIDEr computation
```bash
make compute_ciderdf
```

Pre-compute evaluation scores (BLEU_4, CIDEr, METEOR, ROUGE_L) for each caption
```bash
make compute_evalscores
```

## Train/Test ###

```bash
make train [options]
make test [options]
```

Please refer to the Makefile (and opts.py file) for the set of available train/test options

## Examples

Train XE model
```bash
make train GID=0 EXP_NAME=xe FEATS="resnet c3d mfcc category" USE_RL=0 USE_CST=0 USE_MIXER=0 SCB_CAPTIONS=0 LOGLEVEL=DEBUG MAX_EPOCHS=50
```

Train CST_GT_None/WXE model

```bash
make train GID=0 EXP_NAME=WXE FEATS="resnet c3d mfcc category" USE_RL=1 USE_CST=1 USE_MIXER=0 SCB_CAPTIONS=0 LOGLEVEL=DEBUG MAX_EPOCHS=50
```

Train CST_MS_Greedy model (using greedy baseline)

```bash
make train GID=0 EXP_NAME=CST_MS_Greedy FEATS="resnet c3d mfcc category" USE_RL=1 USE_CST=0 SCB_CAPTIONS=0 USE_MIXER=1 MIXER_FROM=1 USE_EOS=1 LOGLEVEL=DEBUG MAX_EPOCHS=200 START_FROM=output/model/WXE
```

Train CST_MS_SCB model (using SCB baseline, where SCB is computed from GT captions)

```
make train GID=0 EXP_NAME=CST_MS_SCB FEATS="resnet c3d mfcc category" USE_RL=1 USE_CST=1 USE_MIXER=1 MIXER_FROM=1 SCB_BASELINE=1 SCB_CAPTIONS=20 USE_EOS=1 LOGLEVEL=DEBUG MAX_EPOCHS=200 START_FROM=output/model/WXE
```

Train CST_MS_SCB(*) model (using SCB baseline, where SCB is computed from model sampled captions)

```
make train GID=0 MODEL_TYPE=concat EXP_NAME=CST_MS_SCBSTAR FEATS="resnet c3d mfcc category" USE_RL=1 USE_CST=1 USE_MIXER=1 MIXER_FROM=1 SCB_BASELINE=2 SCB_CAPTIONS=20 USE_EOS=1 LOGLEVEL=DEBUG MAX_EPOCHS=200 START_FROM=output/model/WXE
```

If you want to change the input features, modify the `FEATS` variable in above commands.

## Reference

    @article{cst_phan2017,
        author = {Sang Phan and Gustav Eje Henter and Yusuke Miyao and Shin'ichi Satoh},
        title = {Consensus-based Sequence Training for Video Captioning},
        journal = {ArXiv e-prints},
        archivePrefix = "arXiv",
        eprint = {1712.09532},
        year = {2017},
    }
    
## Todo 

* Test on Youtube2Text dataset (different number of captions per video)

### Acknowledgements ###

* Torch implementation of [NeuralTalk2](https://github.com/karpathy/neuraltalk2)
* PyTorch implementation of Self-critical Sequence Training for Image Captioning [(SCST)](https://github.com/ruotianluo/self-critical.pytorch)
* PyTorch Team
