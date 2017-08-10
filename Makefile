
### Directory Setting
IN_DIR=input
OUT_DIR=output
META_DIR=$(OUT_DIR)/metadata

MSRVTT2016_DIR=$(IN_DIR)/msrvtt2016
MSRVTT2017_DIR=$(IN_DIR)/msrvtt2017
YT2T_DIR=$(IN_DIR)/yt2t

SPLITS=train val test
DATASETS=yt2t msrvtt2016 msrvtt2017 

WORD_COUNT_THRESHOLD?=3  # in output/metadata this threshold was 0; was 3 in output/metadata2017
MAX_SEQ_LEN?=30          # in output/metadata seqlen was 20; was 30 in output/metadata2017

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

GID?=7

