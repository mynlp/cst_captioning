
### Directory Setting
IN_DIR=input
OUT_DIR=output
META_DIR=$(OUT_DIR)/metadata
FEAT_DIR=$(OUT_DIR)/feature

MSRVTT2016_DIR=$(IN_DIR)/msrvtt2016
MSRVTT2017_DIR=$(IN_DIR)/msrvtt2017
YT2T_DIR=$(IN_DIR)/yt2t

SPLITS=train val test
DATASETS=yt2t msrvtt2016 msrvtt2017 

WORD_COUNT_THRESHOLD?=3  # in output/metadata this threshold was 0; was 3 in output/metadata2017
MAX_SEQ_LEN?=30          # in output/metadata seqlen was 20; was 30 in output/metadata2017

GID?=3

DATASET?=yt2t
TRAIN_DATASET?=$(DATASET)
VAL_DATASET?=$(DATASET)
TEST_DATASET?=$(DATASET)
TRAIN_SPLIT?=train
VAL_SPLIT?=val
TEST_SPLIT?=test
DATA_ID=$(TRAIN_DATASET)$(TRAIN_SPLIT)_$(VAL_DATASET)$(VAL_SPLIT)_$(TEST_DATASET)$(TEST_SPLIT)
TRAIN_DATA_ID=$(TRAIN_DATASET)$(TRAIN_SPLIT)_$(VAL_DATASET)$(VAL_SPLIT)

LEARNING_RATE?=0.001
BATCH_SIZE?=64
TEST_BATCH_SIZE?=64
TRAIN_SEQ_PER_IMG?=20
TEST_SEQ_PER_IMG?=20
RNN_SIZE?=512
TEST_ONLY?=0

MAX_PATIENCE?=20 # FOR EARLY STOPPING
SAVE_CHECKPOINT_FROM=10
FEAT_SET=c3d

MAX_ITERS?=20000
NUM_CHUNKS?=1
START_FROM?=''
COMBINATION_TYPE=concat
OUTPUT_ATTENTION=0
RESUME?=0
USE_ATTENTION?=0
PRINT_ATT_COEF?=0
ATTENTION_TYPE?=B
BEAM_SIZE?=5
ALIGN_TYPE?=manet2

TODAY=20170714
VER?=exp_$(DATASET)_$(TODAY)_$(COMBINATION_TYPE)_$(ATTENTION_TYPE)_$(ALIGN_TYPE)
VAL_LANG_EVAL?=1
TEST_LANG_EVAL?=1
COMPARE_PPL?=1

POOLING?=mp
CAT_TYPE=glove
DEBUG?=0
USE_SS?=0
USE_ROBUST?=0

FEAT1?=resnet
FEAT2?=c3d
FEAT3?=mfcc
FEAT4?=category
FEAT5?=vgg16
FEAT6?=vgg19

TRAIN_ID=$(TRAIN_DATA_ID)_$(NUM_CHUNKS)_$(USE_ATTENTION)_$(BATCH_SIZE)_$(LEARNING_RATE)_ss$(USE_SS)_robust$(USE_ROBUST)_resume$(RESUME)

##########################################################################

### Standalize data
standalize_datainfo: $(foreach d,$(DATASETS),$(patsubst %,$(META_DIR)/$(d)_%_datainfo.json,$(SPLITS)))
$(META_DIR)/msrvtt2016_%_datainfo.json: $(MSRVTT2016_DIR)/%_videodatainfo.json
	python standalize_format.py $^ $@ --dataset msrvtt2016 --split $*
$(META_DIR)/msrvtt2017_%_datainfo.json: $(MSRVTT2017_DIR)/msrvtt2017_%_videodatainfo.json
	python standalize_format.py $^ $@ --dataset msrvtt2017 --split $* \
		--val2016_json $(MSRVTT2016_DIR)/val_videodatainfo.json 
$(META_DIR)/yt2t_%_datainfo.json: $(YT2T_DIR)/naacl15/sents_%_lc_nopunc.txt
	python standalize_format.py $^ $@ --dataset yt2t

preprocess_datainfo: $(foreach s,$(SPLITS),$(patsubst %,$(META_DIR)/%_$(s)_proprocessedtokens.json,$(DATASETS)))
%_proprocessedtokens.json: %_datainfo.json 
		python preprocess_datainfo.py $^ $@

build_vocab: $(patsubst %,$(META_DIR)/%_train_vocab.json,$(DATASETS))
%_train_vocab.json: %_train_proprocessedtokens.json
		python build_vocab.py $< $@ --word_count_threshold $(WORD_COUNT_THRESHOLD)

create_sequencelabel: $(foreach s,$(SPLITS),$(patsubst %,$(META_DIR)/%_$(s)_sequencelabel.h5,$(DATASETS)))
.SECONDEXPANSION:
%_sequencelabel.h5: $$(firstword $$(subst _, ,$$@))_train_vocab.json %_proprocessedtokens.json
	python create_sequencelabel.py $^ $@ --max_length $(MAX_SEQ_LEN)

## Convert standalized datainfo to coco format for language evaluation
convert_datainfo2cocofmt: $(foreach s,$(SPLITS),$(patsubst %,$(META_DIR)/%_$(s)_cocofmt.json,$(DATASETS)))
%_cocofmt.json: %_datainfo.json 
	python convert_datainfo2cocofmt.py $< $@ 
    
train_single: $(patsubst %,train_single_%,$(FEAT_SET))
train_single_%: $(META_DIR)/$(TRAIN_DATASET)_$(TRAIN_SPLIT)_sequencelabel.h5 \
	$(FEAT_DIR)/$(TRAIN_DATASET)_$(TRAIN_SPLIT)_%_$(POOLING)$(NUM_CHUNKS).h5 \
	$(META_DIR)/$(VAL_DATASET)_$(VAL_SPLIT)_sequencelabel.h5 \
	$(FEAT_DIR)/$(VAL_DATASET)_$(VAL_SPLIT)_%_$(POOLING)$(NUM_CHUNKS).h5 \
	$(META_DIR)/$(VAL_DATASET)_$(VAL_SPLIT)_cocofmt.json \
	$(META_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_sequencelabel.h5 \
	$(FEAT_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_%_$(POOLING)$(NUM_CHUNKS).h5 \
	$(META_DIR)/$(TEST_DATASET)_$(TEST_SPLIT)_cocofmt.json 
	CUDA_VISIBLE_DEVICES=$(GID) python train.py \
		--train_label_h5 $(word 1,$^) \
		--train_feat_h5 $(word 2,$^) \
		--val_label_h5 $(word 3,$^) \
		--val_feat_h5 $(word 4,$^) \
		--val_gold_ann_file $(word 5,$^) \
		--test_label_h5 $(word 6,$^) \
		--test_feat_h5 $(word 7,$^) \
		--test_gold_ann_file $(word 8,$^) \
		--beam_size $(BEAM_SIZE) --max_patience $(MAX_PATIENCE) --compare_ppl $(COMPARE_PPL) \
		--use_ss $(USE_SS) --use_robust $(USE_ROBUST) \
		--language_eval $(VAL_LANG_EVAL) --checkpoint_path $(MODEL_DIR)/$(VER) --max_iters $(MAX_ITERS) --rnn_size $(RNN_SIZE) \
		--train_seq_per_img $(TRAIN_SEQ_PER_IMG) --test_seq_per_img $(TEST_SEQ_PER_IMG) \
		--batch_size $(BATCH_SIZE) --test_batch_size $(TEST_BATCH_SIZE) --learning_rate $(LEARNING_RATE) \
		--save_checkpoint_from $(SAVE_CHECKPOINT_FROM) --num_chunks $(NUM_CHUNKS) --debug $(DEBUG) --combination_type $(COMBINATION_TYPE)\
		--use_attention $(USE_ATTENTION) --att_type $(ATTENTION_TYPE) --align_type $(ALIGN_TYPE) --output_attention $(OUTPUT_ATTENTION) \
		--val_id $(VAL_DATASET)$(VAL_SPLIT) --test_id $(TEST_DATASET)$(TEST_SPLIT) --id $*_$(TRAIN_ID) \
		--test_only $(TEST_ONLY) --resume $(RESUME) 
