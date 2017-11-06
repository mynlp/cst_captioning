
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
DATASETS=yt2t msrvtt# msrvtt2017 tvvtt

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

LEARNING_RATE?=0.0001
LR_UPDATE?=200
BATCH_SIZE?=64
TRAIN_SEQ_PER_IMG?=20
TEST_SEQ_PER_IMG?=20
RNN_SIZE?=512

PRINT_INTERVAL?=20
MAX_PATIENCE?=5 # FOR EARLY STOPPING
SAVE_CHECKPOINT_FROM?=10
FEAT_SET=c3d

MAX_EPOCHS?=200
NUM_CHUNKS?=1
PRINT_ATT_COEF?=0
BEAM_SIZE?=5

TODAY=20170831
EXP_NAME?=exp_$(DATASET)_$(TODAY)
VAL_LANG_EVAL?=1
TEST_LANG_EVAL?=1
EVAL_METRIC?=CIDEr
START_FROM?=No
MODEL_TYPE?=concat
POOLING?=mp
CAT_TYPE=glove
LOGLEVEL?=INFO
USE_SS?=0
USE_SS_AFTER?=5
SS_MAX_PROB?=0.25
USE_ROBUST?=0
NUM_ROBUST?=0
R_BASELINE?=1
USE_SCST?=0
USE_MIXER?=0
SS_K?=100


FEAT1?=resnet
FEAT2?=c3d
FEAT3?=mfcc
FEAT4?=category
FEAT5?=vgg16
FEAT6?=vgg19

#FEATS=$(FEAT1) $(FEAT2) $(FEAT3) $(FEAT5) $(FEAT6)
FEATS=$(FEAT1) $(FEAT2) $(FEAT6)

TRAIN_ID=$(TRAIN_DATASET)_$(MODEL_TYPE)_$(EVAL_METRIC)_$(BATCH_SIZE)_$(LEARNING_RATE)

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

# newer version: use vocab of all words, rather than with word freq > 3
prepro_cidercache: $(foreach s,$(SPLITS),$(patsubst %,$(META_DIR)/%_$(s)_cidercacheall.pkl,$(DATASETS)))
%_cidercacheall.pkl: %_proprocessedtokens.json
	python prepro_ngrams.py $^ $@ --output_words
#####################################################################################################################

noop=
space=$(noop) $(noop)

TRAIN_OPT=--beam_size $(BEAM_SIZE) --max_patience $(MAX_PATIENCE) --eval_metric $(EVAL_METRIC) --print_log_interval $(PRINT_INTERVAL)\
	--language_eval $(VAL_LANG_EVAL) --max_epochs $(MAX_EPOCHS) --rnn_size $(RNN_SIZE) \
	--train_seq_per_img $(TRAIN_SEQ_PER_IMG) --test_seq_per_img $(TEST_SEQ_PER_IMG) \
	--batch_size $(BATCH_SIZE) --test_batch_size $(BATCH_SIZE) --learning_rate $(LEARNING_RATE) --lr_update $(LR_UPDATE) \
	--save_checkpoint_from $(SAVE_CHECKPOINT_FROM) --num_chunks $(NUM_CHUNKS) \
	--train_cached_tokens $(META_DIR)/$(TRAIN_DATASET)_train_cidercache.pkl \
	--use_ss $(USE_SS) --ss_k $(SS_K) --use_scst_after $(USE_SS_AFTER) --ss_max_prob $(SS_MAX_PROB) \
	--use_scst $(USE_SCST) --use_mixer $(USE_MIXER) \
	--use_robust $(USE_ROBUST) --num_robust $(NUM_ROBUST) --use_robust_baseline $(R_BASELINE) \
	--loglevel $(LOGLEVEL) --model_type $(MODEL_TYPE) \
	--model_file $@ --start_from $(START_FROM) --result_file $(basename $@)_test.json \
	2>&1 | tee $(basename $@).log

TEST_OPT=--beam_size $(BEAM_SIZE) \
	--language_eval $(VAL_LANG_EVAL) \
	--test_seq_per_img $(TEST_SEQ_PER_IMG) \
	--test_batch_size $(BATCH_SIZE) \
	--loglevel $(LOGLEVEL) \
	--result_file $@

train: $(patsubst %,$(MODEL_DIR)/$(EXP_NAME)/%_$(TRAIN_ID).pth,$(FEATS))
$(MODEL_DIR)/$(EXP_NAME)/%_$(TRAIN_ID).pth: \
	$(META_DIR)/$(TRAIN_DATASET)_$(TRAIN_SPLIT)_sequencelabel.h5 \
	$(META_DIR)/$(VAL_DATASET)_$(VAL_SPLIT)_sequencelabel.h5 \
	$(META_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_sequencelabel.h5 \
	$(META_DIR)/$(TRAIN_DATASET)_$(TRAIN_SPLIT)_cocofmt.json \
	$(META_DIR)/$(VAL_DATASET)_$(VAL_SPLIT)_cocofmt.json \
	$(META_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_cocofmt.json \
        $(FEAT_DIR)/$(TRAIN_DATASET)_$(TRAIN_SPLIT)_%_mp$(NUM_CHUNKS).h5 \
	$(FEAT_DIR)/$(VAL_DATASET)_$(VAL_SPLIT)_%_mp$(NUM_CHUNKS).h5 \
	$(FEAT_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_%_mp$(NUM_CHUNKS).h5 
	mkdir -p $(MODEL_DIR)/$(EXP_NAME)
	CUDA_VISIBLE_DEVICES=$(GID) python train.py \
		--train_label_h5 $(word 1,$^) \
		--val_label_h5 $(word 2,$^) \
		--test_label_h5 $(word 3,$^) \
		--train_cocofmt_file $(word 4,$^) \
		--val_cocofmt_file $(word 5,$^) \
		--test_cocofmt_file $(word 6,$^) \
		--train_feat_h5 $(word 7,$^) \
		--val_feat_h5 $(word 8,$^) \
		--test_feat_h5 $(word 9,$^) \
		$(TRAIN_OPT)

test: $(patsubst %,$(MODEL_DIR)/$(EXP_NAME)/%_$(TRAIN_ID)_test.json,$(FEATS))
$(MODEL_DIR)/$(EXP_NAME)/%_$(TRAIN_ID)_test.json: \
	$(MODEL_DIR)/$(EXP_NAME)/%_$(TRAIN_ID).pth \
	$(META_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_sequencelabel.h5 \
	$(META_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_cocofmt.json \
	$(FEAT_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_%_mp$(NUM_CHUNKS).h5
	CUDA_VISIBLE_DEVICES=$(GID) python test.py \
		--model_file $(word 1,$^) \
		--test_label_h5 $(word 2,$^) \
		--test_cocofmt_file $(word 3,$^) \
		--test_feat_h5 $(word 4,$^) \
		$(TEST_OPT)

train_multimodal: $(MODEL_DIR)/$(EXP_NAME)/$(subst $(space),$(noop),$(FEATS))_$(TRAIN_ID).pth
$(MODEL_DIR)/$(EXP_NAME)/$(subst $(space),$(noop),$(FEATS))_$(TRAIN_ID).pth: \
	$(META_DIR)/$(TRAIN_DATASET)_$(TRAIN_SPLIT)_sequencelabel.h5 \
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
		$(TRAIN_OPT)

test_multimodal: $(MODEL_DIR)/$(EXP_NAME)/$(subst $(space),$(noop),$(FEATS))_$(TRAIN_ID)_test.json
$(MODEL_DIR)/$(EXP_NAME)/$(subst $(space),$(noop),$(FEATS))_$(TRAIN_ID)_test.json: \
	$(MODEL_DIR)/$(EXP_NAME)/$(subst $(space),$(noop),$(FEATS))_$(TRAIN_ID).pth \
	$(META_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_sequencelabel.h5 \
	$(META_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_cocofmt.json \
	$(patsubst %,$(FEAT_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_%_mp$(NUM_CHUNKS).h5,$(FEATS))
	CUDA_VISIBLE_DEVICES=$(GID) python test.py \
		--model_file $(word 1,$^) \
		--test_label_h5 $(word 2,$^) \
		--test_cocofmt_file $(word 3,$^) \
		--test_feat_h5 $(patsubst %,$(FEAT_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_%_mp$(NUM_CHUNKS).h5,$(FEATS))\
		$(TEST_OPT)


compute_ciderscores: $(patsubst %,$(META_DIR)/$(TRAIN_DATASET)_%_ciderscores.pkl,$(SPLITS))
%_ciderscores.pkl: %_cocofmt.json
	python compute_cider.py $^ $@

compute_dataslice:
	python compute_dataslice.py \
		$(MODEL_DIR)/$(EXP_NAME)/$(subst $(space),$(noop),$(FEATS))_$(TRAIN_ID)_test.json \
		$(META_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_cocofmt.json \
		$(META_DIR)/$(TRAIN_DATASET)_test_ciderscores.pkl rl.txt

compute_ciderd:
	python compute_ciderd.py \
		$(MODEL_DIR)/xe_robust_r0/resnetc3dmfcccategory_msrvtt_concat_CIDEr_64_0.0001_test.json \
		$(META_DIR)/$(TRAIN_DATASET)_test_cidercacheall_words.pkl \
		$(META_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_cocofmt.json 
	
# If you want all intermediates to remain
# .SECONDARY:

# You can use the wildcard with .PRECIOUS.
.PRECIOUS: %.pth

