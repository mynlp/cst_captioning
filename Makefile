
### Directory Setting
IN_DIR=input
OUT_DIR=output
META_DIR=$(OUT_DIR)/metadata
FEAT_DIR=$(OUT_DIR)/feature
MODEL_DIR=$(OUT_DIR)/model

MSRVTT2016_DIR=$(IN_DIR)/msrvtt
MSRVTT2017_DIR=$(IN_DIR)/msrvtt2017
YT2T_DIR=$(IN_DIR)/yt2t

SPLITS=train val test
DATASETS=yt2t msrvtt msrvtt2017 tvvtt

WORD_COUNT_THRESHOLD?=3  # in output/metadata this threshold was 0; was 3 in output/metadata2017
MAX_SEQ_LEN?=30          # in output/metadata seqlen was 20; was 30 in output/metadata2017

GID?=5

DATASET?=msrvtt
TRAIN_DATASET?=$(DATASET)
VAL_DATASET?=$(DATASET)
TEST_DATASET?=$(DATASET)
TRAIN_SPLIT?=train
VAL_SPLIT?=val
TEST_SPLIT?=test
DATA_ID=$(TRAIN_DATASET)$(TRAIN_SPLIT)_$(VAL_DATASET)$(VAL_SPLIT)_$(TEST_DATASET)$(TEST_SPLIT)

LEARNING_RATE?=0.0001
BATCH_SIZE?=128
TEST_BATCH_SIZE?=128
TRAIN_SEQ_PER_IMG?=20
TEST_SEQ_PER_IMG?=20
RNN_SIZE?=512
TEST_ONLY?=0

MAX_PATIENCE?=5 # FOR EARLY STOPPING
SAVE_CHECKPOINT_FROM?=20
FEAT_SET=c3d

MAX_ITERS?=20000
NUM_CHUNKS?=1
PRINT_ATT_COEF?=0
BEAM_SIZE?=5

TODAY=20170831
EXP_NAME?=exp_$(DATASET)_$(TODAY)
VAL_LANG_EVAL?=1
TEST_LANG_EVAL?=1
COMPARE_PPL?=1

MODEL_TYPE?=standard
POOLING?=mp
CAT_TYPE=glove
LOGLEVEL?=INFO
USE_SS?=0
USE_ROBUST?=0
USE_SCST?=0

FEAT1?=resnet
FEAT2?=c3d
FEAT3?=mfcc
FEAT4?=category
FEAT5?=vgg16
FEAT6?=vgg19

#FEATS=$(FEAT1) $(FEAT2) $(FEAT3) $(FEAT5) $(FEAT6)
FEATS=$(FEAT1)

TRAIN_ID=$(DATA_ID)_$(MODEL_TYPE)_$(NUM_CHUNKS)_$(BATCH_SIZE)_$(LEARNING_RATE)

###################################################################################################################
###
pre_process: standalize_datainfo preprocess_datainfo build_vocab create_sequencelabel convert_datainfo2cocofmt

### Standalize data
standalize_datainfo: $(foreach d,$(DATASETS),$(patsubst %,$(META_DIR)/$(d)_%_datainfo.json,$(SPLITS)))
$(META_DIR)/msrvtt_%_datainfo.json: $(MSRVTT2016_DIR)/%_videodatainfo.json
	python standalize_format.py $^ $@ --dataset msrvtt2016 --split $*
$(META_DIR)/msrvtt2017_%_datainfo.json: $(MSRVTT2017_DIR)/msrvtt2017_%_videodatainfo.json
	python standalize_format.py $^ $@ --dataset msrvtt2017 --split $* \
		--val2016_json $(MSRVTT2016_DIR)/val_videodatainfo.json 
$(META_DIR)/yt2t_%_datainfo.json: $(YT2T_DIR)/naacl15/sents_%_lc_nopunc.txt
	python standalize_format.py $^ $@ --dataset yt2t
$(META_DIR)/tvvtt_%_datainfo.json: $(META_DIR)/v2t2017_infos.json 
	python standalize_format.py $^ $@ --dataset tvvtt --split $*
### 
preprocess_datainfo: $(foreach s,$(SPLITS),$(patsubst %,$(META_DIR)/%_$(s)_proprocessedtokens.json,$(DATASETS)))
%_proprocessedtokens.json: %_datainfo.json 
		python preprocess_datainfo.py $^ $@

###
build_vocab: $(patsubst %,$(META_DIR)/%_train_vocab.json,$(DATASETS))
%_train_vocab.json: %_train_proprocessedtokens.json
		python build_vocab.py $< $@ --word_count_threshold $(WORD_COUNT_THRESHOLD)
###
create_sequencelabel: $(foreach s,$(SPLITS),$(patsubst %,$(META_DIR)/%_$(s)_sequencelabel.h5,$(DATASETS)))
.SECONDEXPANSION:
%_sequencelabel.h5: $$(firstword $$(subst _, ,$$@))_train_vocab.json %_proprocessedtokens.json
	python create_sequencelabel.py $^ $@ --max_length $(MAX_SEQ_LEN)

### Convert standalized datainfo to coco format for language evaluation
convert_datainfo2cocofmt: $(foreach s,$(SPLITS),$(patsubst %,$(META_DIR)/%_$(s)_cocofmt.json,$(DATASETS)))
%_cocofmt.json: %_datainfo.json 
	python convert_datainfo2cocofmt.py $< $@ 

### frameinfo
get_dataset = $(word 1,$(subst _, ,$1))
get_split = $(word 2,$(subst _, ,$1))
create_frameinfo: $(foreach s,$(SPLITS),$(patsubst %,$(META_DIR)/%_$(s)_frameinfo.json,$(DATASETS)))
%_frameinfo.json: %_datainfo.json 
	python create_frameinfo.py $^ $@ \
	       --dataset $(call get_dataset,$(notdir $@)) \
	       --split $(call get_split,$(notdir $@)) \
	       --input_dir $(IN_DIR) --img_type rgb

### create cached of document frequency for Cider computation
prepro_ngrams: $(foreach s,$(SPLITS),$(patsubst %,$(META_DIR)/%_$(s)_cidercache.pkl,$(DATASETS)))
.SECONDEXPANSION:
%_cidercache.pkl: $$(firstword $$(subst _, ,$$@))_train_vocab.json %_proprocessedtokens.json
	python prepro_ngrams.py $^ $@ --output_words

#####################################################################################################################

noop=
space=$(noop) $(noop)

train: $(META_DIR)/$(TRAIN_DATASET)_$(TRAIN_SPLIT)_sequencelabel.h5 \
	$(META_DIR)/$(VAL_DATASET)_$(VAL_SPLIT)_sequencelabel.h5 \
	$(META_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_sequencelabel.h5 \
	$(META_DIR)/$(TRAIN_DATASET)_$(TRAIN_SPLIT)_cocofmt.json \
	$(META_DIR)/$(VAL_DATASET)_$(VAL_SPLIT)_cocofmt.json \
	$(META_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_cocofmt.json \
        $(patsubst %,$(FEAT_DIR)/$(TRAIN_DATASET)_$(TRAIN_SPLIT)_%_mp$(NUM_CHUNKS).h5,$(FEATS)) \
	$(patsubst %,$(FEAT_DIR)/$(VAL_DATASET)_$(VAL_SPLIT)_%_mp$(NUM_CHUNKS).h5,$(FEATS)) \
	$(patsubst %,$(FEAT_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_%_mp$(NUM_CHUNKS).h5,$(FEATS))
	mkdir -p $(MODEL_DIR)/$(EXP_NAME)
	CUDA_VISIBLE_DEVICES=$(GID) python train.py \
		--train_label_h5 $(word 1,$^) \
		--val_label_h5 $(word 2,$^) \
		--test_label_h5 $(word 3,$^) \
		--train_cocofmt_file $(word 4,$^) \
		--val_cocofmt_file $(word 5,$^) \
		--test_cocofmt_file $(word 6,$^) \
		--train_feat_h5 $(patsubst %,$(FEAT_DIR)/$(TRAIN_DATASET)_$(TRAIN_SPLIT)_%_mp$(NUM_CHUNKS).h5,$(FEATS))\
		--val_feat_h5 $(patsubst %,$(FEAT_DIR)/$(VAL_DATASET)_$(VAL_SPLIT)_%_mp$(NUM_CHUNKS).h5,$(FEATS))\
		--test_feat_h5 $(patsubst %,$(FEAT_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_%_mp$(NUM_CHUNKS).h5,$(FEATS))\
		--beam_size $(BEAM_SIZE) --max_patience $(MAX_PATIENCE) --compare_ppl $(COMPARE_PPL) --eval_metrics Loss \
		--language_eval $(VAL_LANG_EVAL) --checkpoint_path $(MODEL_DIR)/$(EXP_NAME) --max_iters $(MAX_ITERS) --rnn_size $(RNN_SIZE) \
		--train_seq_per_img $(TRAIN_SEQ_PER_IMG) --test_seq_per_img $(TEST_SEQ_PER_IMG) \
		--batch_size $(BATCH_SIZE) --test_batch_size $(BATCH_SIZE) --learning_rate $(LEARNING_RATE) \
		--save_checkpoint_from $(SAVE_CHECKPOINT_FROM) --num_chunks $(NUM_CHUNKS) \
		--test_only $(TEST_ONLY) \
		--id $(subst $(space),$(noop),$(FEATS))_$(TRAIN_ID) \
		--train_cached_tokens $(META_DIR)/$(TRAIN_DATASET)_train_cidercache.pkl --use_scst_after 30 --use_scst $(USE_SCST) \
		--loglevel $(LOGLEVEL) --model_type $(MODEL_TYPE) 
		#  2>&1 | tee $(MODEL_DIR)/$(EXP_NAME)/$(subst $(space),$(noop),$(FEATS))_$(TRAIN_ID).log 

###########################

METEOR_TOOL=/home/plsang/works/tools/meteor-1.5
#RUN_FILE=/home/plsang/works/tvv2t/output/submission/description/resnet.txt
RUN_FILE=/home/plsang/works/captioning.pytorch/output/model/exp_msrvtt2017_20170830/resnetc3dmfccvgg16vgg19_msrvtt2017train_msrvtt2017val_tvvttval_1_64_0.001_ss0_robust0_Loss_test_predictions.txt

GT16_FILE=/home/plsang/works/v2t2017/input/v2t2016/tv16.ref.meteor
test:
	java -Xmx2G -jar $(METEOR_TOOL)/meteor-1.5.jar $(RUN_FILE) $(GT16_FILE) -l en -norm -r 2 -t adq

RUN1_FILE=$(MODEL_DIR)/$(EXP_NAME)/resnetc3dmfccvgg16vgg19_msrvtt2017train_msrvtt2017val_tvvtttest_manet_1_64_0.001_Loss_test_predictions.json
RUN2_FILE=$(MODEL_DIR)/$(EXP_NAME)/resnetc3dmfccvgg16vgg19_msrvtt2017train_msrvtt2017val_tvvtttest_concat_1_64_0.001_Loss_test_predictions.json
VTT2017_SUBMIT_DIR=/home/plsang/active/v2t2017/output/result_to_submit/captioning
convert_v2t2017:
	python convert_coco2trecvidfmt.py $(RUN1_FILE) $(VTT2017_SUBMIT_DIR)/NII_Hitachi_UIT_R1_primary.txt 
	python convert_coco2trecvidfmt.py $(RUN2_FILE) $(VTT2017_SUBMIT_DIR)/NII_Hitachi_UIT_R2.txt 

