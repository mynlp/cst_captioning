# Consensus-based Sequence Training for Video Captioning #

## Dependencies ###

* Python 2.7
* Pytorch 0.2
* [Microsoft COCO Caption Evaluation](https://github.com/tylin/coco-caption)
* [CIDEr](https://github.com/ruotianluo/cider)

(Check out the `coco-caption` and `cider` projects into your working directory)

## Getting started ###

* Extract video features:
  - Extracted features can be shared

* Generate metadata

```bash
make pre_process
```
this will run `standalize_datainfo` `preprocess_datainfo` `build_vocab` `create_sequencelabel` `convert_datainfo2cocofmt`

* Create cached of document frequency for CIDEr computation
```bash
make prepro_cidercache
```

* Pre-compute evaluation scores (BLEU_4, CIDEr, METEOR, ROUGE_L) for each caption
```bash
make compute_evalscores
```

## Train/Test ###

* Train/test single model
```bash
make train
make test
```
* Train/test fusion model (multimodal features)
```bash
make train_multimodal
make test_multimodal
```

## Examples of using make rules for training:
* Cross-entropy Training (XE)
```bash
make train_multimodal GID=0 MODEL_TYPE=concat EXP_NAME=XE FEATS="resnet c3d mfcc category" USE_SS_AFTER=0 USE_ROBUST=0 USE_MIXER=0 NUM_ROBUST=0 USE_SCST=0 SS_K=0 LOGLEVEL=DEBUG MAX_EPOCHS=50
```

* CST on ground-truth data (CST_GT_None/WXE)

```bash
make train_multimodal GID=0 MODEL_TYPE=concat EXP_NAME=WXE FEATS="resnet c3d mfcc category" USE_SS_AFTER=0 USE_ROBUST=1 USE_MIXER=0 NUM_ROBUST=0 MIXER_FROM=0 USE_SCST=1 SS_K=0 LOGLEVEL=DEBUG MAX_EPOCHS=50
```

* CST on model sampled data using the greedy baseline (CST_MS_Greedy)

```bash
make train_multimodal GID=0 MODEL_TYPE=concat EXP_NAME=CST_MS_Greedy FEATS="resnet c3d mfcc category" USE_SS_AFTER=0 USE_ROBUST=0 NUM_ROBUST=0 USE_MIXER=1 MIXER_FROM=1 USE_SCST=1 USE_EOS=1 START_FROM=output/model/cvpr2018_cstxe LOGLEVEL=DEBUG MAX_EPOCHS=200
```

* CST on model sampled data using the self-consensus baseline from the GT (CST_MS_SCB)

```
make train_multimodal GID=0 MODEL_TYPE=concat EXP_NAME=CST_MS_SCB FEATS="resnet c3d mfcc category" USE_SS_AFTER=0 USE_ROBUST=1 USE_MIXER=1 MIXER_FROM=1 R_BASELINE=2 NUM_ROBUST=20 USE_SCST=1 USE_EOS=1 START_FROM=output/model/cvpr2018_cstxe LOGLEVEL=DEBUG MAX_EPOCHS=200
```

* CST on model sampled data using the self-consensus baseline from the sampled sequences (CST_MS_SCB(*))

```
make train_multimodal GID=0 MODEL_TYPE=concat EXP_NAME=CST_MS_SCB FEATS="resnet c3d mfcc category" USE_SS_AFTER=0 USE_ROBUST=1 USE_MIXER=1 MIXER_FROM=1 R_BASELINE=1 NUM_ROBUST=20 USE_SCST=1 USE_EOS=1 START_FROM=output/model/cvpr2018_cstxe LOGLEVEL=DEBUG MAX_EPOCHS=200
```

## Overview of training methods

| Label                                          | Start from | Sequences used in training | Weighted?   | Baseline $b$                                                         | Comment                                               |
|------------------------------------------------|------------|----------------------------|-------------|----------------------------------------------------------------------|-------------------------------------------------------|
| XE                                             | Scratch    | Ground truth               | No (not RL) | None                                                                 | Unweighted log-likelihood; bottom line                |
| CST\_GT\_None                                  | Scratch    | Ground truth (``GT'')      | Yes         | None                                                                 | Weighted log-likelihood; a.k.a.\textbackslash ``WXE'' |
| CST\_MS\_Greedy                                | WXE        | Model samples (``MS'')     | Yes         | Greedy$\sim$\cite\{Rennie2017CVPR\}                                  |                                                       |
| CST\_MS\_SCB                                   | WXE        | Model samples              | Yes         | SCB                                                                  | ``Full CST''                                          |
| CIDEnt-RL$\sim$\cite\{pasunuru2017reinforced\} | XE         | Truth \& samples (MIXER)   | Yes         | Lin.\textbackslash reg.\textbackslash $\sim$\cite\{Ranzato2016ICLR\} | Previous best                                         |
| SCST$\sim$\cite\{Rennie2017CVPR\}              | XE         | Model samples              | Yes         | Greedy                                                               | Our implementation                                    |

## Reference

    @article{cst_phan2017,
        author = {Sang Phan and Gustav Eje Henter and Yusuke Miyao and Shin’ichi Satoh},
        title = {Consensus-based Sequence Training for Video Captioning},
        booktitle = {arXiv},
        year = {2017},
    }
    
## TODO 

* Test on Youtube2Text dataset (different number of captions per video)
* Clean code
* Share features

### Acknowledgements ###

* Torch implementation of [NeuralTalk2](https://github.com/karpathy/neuraltalk2)
* PyTorch implementation of Self-critical Sequence Training for Image Captioning [(SCST)](https://github.com/ruotianluo/self-critical.pytorch)
* PyTorch Team
