
GID?=7

train_a:
	CUDA_VISIBLE_DEVICES=$(GID) python train.py --batch_size 16 --model_path models_a \
			    2>&1 | tee log/modela.log
train_b:
	CUDA_VISIBLE_DEVICES=$(GID) python train.py --batch_size 16 --model_path models_b \
			    2>&1 | tee log/modelb.log
