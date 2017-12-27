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
