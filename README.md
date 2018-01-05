# Consensus-based Sequence Training for Video Captioning #

## Dependencies ###

* Python 2.7
* Pytorch 0.2
* [Microsoft COCO Caption Evaluation](https://github.com/tylin/coco-caption)
* [CIDEr](https://github.com/ruotianluo/cider)

(Check out the `coco-caption` and `cider` projects into your working directory)

## Getting started ###

Extract video features:
  - Extracted features can be shared

Generate metadata

```bash
make pre_process
```
this will run `standalize_datainfo` `preprocess_datainfo` `build_vocab` `create_sequencelabel` `convert_datainfo2cocofmt`

Create cached of document frequency for CIDEr computation
```bash
make compute_ciderdf
```

Pre-compute evaluation scores (BLEU_4, CIDEr, METEOR, ROUGE_L) for each caption
```bash
make compute_evalscores
```

## Train/Test ###

* Train/test single model
```bash
make train [options]
make test [options]
```
* Train/test fusion model (multimodal features)
```bash
make train_multimodal [options]
make test_multimodal [options]
```

Please refer to the Makefile and the opts.py file for the set of available train/test options

## Examples of using make rules for training:
* Cross-entropy Training (XE)
```bash
make train_multimodal GID=0 MODEL_TYPE=concat EXP_NAME=XE \
FEATS="resnet c3d mfcc category" USE_SS_AFTER=0 USE_ROBUST=0 \
USE_MIXER=0 NUM_ROBUST=0 USE_SCST=0 SS_K=0 LOGLEVEL=DEBUG MAX_EPOCHS=50
```

* CST on ground-truth data (CST_GT_None/WXE)

```bash
make train_multimodal GID=0 MODEL_TYPE=concat EXP_NAME=WXE \
FEATS="resnet c3d mfcc category" USE_SS_AFTER=0 USE_ROBUST=1 \
USE_MIXER=0 NUM_ROBUST=0 MIXER_FROM=0 USE_SCST=1 SS_K=0 LOGLEVEL=DEBUG MAX_EPOCHS=50
```

* CST on model sampled data using the greedy baseline (CST_MS_Greedy)

```bash
make train_multimodal GID=0 MODEL_TYPE=concat EXP_NAME=CST_MS_Greedy \
FEATS="resnet c3d mfcc category" USE_SS_AFTER=0 USE_ROBUST=0 NUM_ROBUST=0 \
USE_MIXER=1 MIXER_FROM=1 USE_SCST=1 USE_EOS=1 LOGLEVEL=DEBUG MAX_EPOCHS=200 \
START_FROM=output/model/cvpr2018_cstxe
```

* CST on model sampled data using the self-consensus baseline from the GT (CST_MS_SCB)

```
make train_multimodal GID=0 MODEL_TYPE=concat EXP_NAME=CST_MS_SCB \
FEATS="resnet c3d mfcc category" USE_SS_AFTER=0 USE_ROBUST=1 USE_MIXER=1 \
MIXER_FROM=1 R_BASELINE=2 NUM_ROBUST=20 USE_SCST=1 USE_EOS=1 LOGLEVEL=DEBUG MAX_EPOCHS=200 \
START_FROM=output/model/cvpr2018_cstxe
```

* CST on model sampled data using the self-consensus baseline from the sampled sequences (CST_MS_SCB(*))

```
make train_multimodal GID=0 MODEL_TYPE=concat EXP_NAME=CST_MS_SCBSTAR \
FEATS="resnet c3d mfcc category" USE_SS_AFTER=0 USE_ROBUST=1 USE_MIXER=1 \
MIXER_FROM=1 R_BASELINE=1 NUM_ROBUST=20 USE_SCST=1 USE_EOS=1 LOGLEVEL=DEBUG MAX_EPOCHS=200 \
START_FROM=output/model/cvpr2018_cstxe
```

## Reference

    @article{cst_phan2017,
        author = {Sang Phan and Gustav Eje Henter and Yusuke Miyao and Shin'ichi Satoh},
        title = {Consensus-based Sequence Training for Video Captioning},
        booktitle = {arXiv},
        year = {2017},
    }
    
## TODO 

* Test on Youtube2Text dataset (different number of captions per video)
* Clean code
* Share features
* Support training on multi GPUs

### Acknowledgements ###

* Torch implementation of [NeuralTalk2](https://github.com/karpathy/neuraltalk2)
* PyTorch implementation of Self-critical Sequence Training for Image Captioning [(SCST)](https://github.com/ruotianluo/self-critical.pytorch)
* PyTorch Team
