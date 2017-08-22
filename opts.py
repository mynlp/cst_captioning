
import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_label_h5', type=str, help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--val_label_h5', type=str, help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--test_label_h5', type=str, help='path to the h5file containing the preprocessed dataset')
    
    parser.add_argument('--train_feat_h5', type=str, help='path to the h5 file containing extracted features')
    parser.add_argument('--val_feat_h5', type=str, help='path to the h5 file containing extracted features')
    parser.add_argument('--test_feat_h5', type=str, help='path to the h5 file containing extracted features')

    parser.add_argument('--train_gold_ann_file', type=str, help='Gold ann file, path relative to the coco-caption root')
    parser.add_argument('--val_gold_ann_file', type=str, help='Gold ann file, path relative to the coco-caption root')
    parser.add_argument('--test_gold_ann_file', type=str, help='Gold ann file, path relative to the coco-caption root')
    
    # Optimization: General
    parser.add_argument('--max_patience', type=int, default=50, help='max number of epoch to run since the minima is detected -- early stopping')
    parser.add_argument('--batch_size', type=int, default=128, help='Video batch size (there will be x seq_per_img sentences)')
    parser.add_argument('--test_batch_size', type=int, default=32, help='what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
    parser.add_argument('--train_seq_per_img', type=int, default=20, help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive.')
    parser.add_argument('--test_seq_per_img', type=int, default=20, help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--cnn_learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--resume', type=int, default=0, help='path to a model checkpoint to initialize model weights from. Empty = don\'t')
    
    # Model settings
    parser.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'gru', 'rnn'], help= 'type of RNN')
    parser.add_argument('--rnn_size', type=int, default=512, help= 'size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_lm_layer', type=int, default=1, help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--input_encoding_size', type=int, default=512, help='the encoding size of each frame in the video.')
    parser.add_argument('--max_iters', type=int, default=-1, help='max number of iterations to run for (-1 = run forever)')
    parser.add_argument('--max_epochs', type=int, default=-1, help='max number of epochs to run for (-1 = run forever)')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='strength of dropout in the Language Model RNN')

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam', help='what update to use? sgd|sgdmom|adagrad|adam')
    parser.add_argument('--finetune_cnn_after', type=int, default=0, help='After what iteration do we start finetuning the CNN? --> will change to finetune IM layer')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1, help='at what iteration to start decaying learning rate? (-1 = dont)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=10000, help='every how many iterations thereafter to drop LR by half?')
    parser.add_argument('--optim_alpha', type=float, default=0.8, help='alpha for adagrad/rmsprop/momentum/adam')
    parser.add_argument('--optim_beta', type=float, default=0.999, help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8, help='epsilon that goes into denominator for smoothing')
    
    # Evaluation/Checkpointing
    parser.add_argument('--num_val_videos', type=int, default=-1, help='how many videos to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--num_test_videos', type=int, default=-1, help='how many videos to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_from', type=int, default=20, help='Start saving checkpoint from this epoch')
    parser.add_argument('--save_checkpoint_every', type=int, default=5, help='how often to save a model checkpoint in epochs?')
    parser.add_argument('--save_checkpoint_decay_start', type=int, default=-1, help='--1: do not decrease. otherwise, decrease after this number')
    parser.add_argument('--save_checkpoint_decay', type=int, default=200, help='how often to save a model checkpoint?')
    parser.add_argument('--checkpoint_path', type=str, default='output/model', help='folder to save checkpoints into (empty = this folder)')
    parser.add_argument('--language_eval', type=int, default=1, help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--eval_metrics', default=['Loss'], nargs='+', choices=['Loss', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'MSRVTT'], help='Evaluation metrics')
    parser.add_argument('--test_language_eval', type=int, default=1, help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    
    parser.add_argument('--print_log_interval', type=int, default=20, help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--loglevel', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    
    ## misc
    parser.add_argument('--backend', type=str, default='cudnn', help='nn|cudnn')
    parser.add_argument('--id', type=str, default=None, help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--seed', type=int, default=123, help='random number generator seed to use')
    parser.add_argument('--gpuid', type=int, default=7, help='which gpu to use. -1 = use CPU')
    parser.add_argument('--num_chunks', type=int, default=1, help='1: no attention, > 1: attention with num_chunks')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the lstm ')
    parser.add_argument('--use_attention', type=int, default=0, help='0: no attention, 1: attention with num_chunks')
    parser.add_argument('--print_att_coef', type=int, default=0, help='0: no attention, 1: attention with num_chunks')
    parser.add_argument('--output_attention', type=int, default=0, help='0: not adding output attention, 1: adding output attention')
    parser.add_argument('--att_type',  type=str, default='B', help='A/B. A: feed null token at start. B: feed features as start.')
    parser.add_argument('--align_type', type=str, default= 'bilinear', help='sum,concat,dot,bilinear')
    parser.add_argument('--combination_type',  type=str, default='concat', help='sum, concat: how to combine two vectors')
    parser.add_argument('--attention_size', type=int, default=512, help='size of the the attentional input vector')
    parser.add_argument('--adaptive_input_size', type=int, default=0, help='if 1, multiply the image_input_encoding_size by number of features and chunks')
    parser.add_argument('--use_category', type=int, default=0, help='0: no category, 1: concat with image embedding, 2: use as start token, 3: use as start token + finetune')
    parser.add_argument('--num_category', type=int, default=20, help='number of category')
    parser.add_argument('--debug', type=int, default=0, help='debug mode')
    parser.add_argument('--train_top_longest_caption', type=int, default=-1, help='select caption in the top longest captions')
    parser.add_argument('--val_top_longest_caption', type=int, default=-1, help='select caption in the top longest captions')
    parser.add_argument('--test_top_longest_caption', type=int, default=-1, help='select caption in the top longest captions')
    
    # parser.add_argument('--eval_criteria', 'CIDEr', 'criteria to select model during cross validation: loss, CIDEr, METEOR, etc (language score)')
    parser.add_argument('--train_id', type=str, default='msrvtttrain', help='id')
    parser.add_argument('--val_id', type=str, default='msrvttval', help='id')
    parser.add_argument('--test_id',  type=str, default='tvv2tttrain', help='id')
    parser.add_argument('--test_checkpoint',  type=str, default='', help='path to the checkpoint needed to be tested')
    parser.add_argument('--test_only', type=int, default=0, help='1: use the current model (located in current path) for testing')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam search size')
    parser.add_argument('--compare_ppl', type=int, default=1, help='Compare done beams by ppl instead of logprob')

    parser.add_argument('--use_ss', type=int, default=0, help='Use schedule sampling')
    parser.add_argument('--ss_start_epoch', type=int, default=1, help='Use schedule sampling')
    parser.add_argument('--ss_start', type=int, default=1, help='Use schedule sampling')
    parser.add_argument('--ss_end', type=float, default=0.9, help='Use schedule sampling')
    parser.add_argument('--ss_k', type=int, default=100, help='plot k/(k+exp(x/k)) from x=0 to 400, k=100')
    parser.add_argument('--draw_loss', type=int, default=0, help='to draw loss or not')

    parser.add_argument('--use_robust', type=int, default=0, help='Use schedule sampling')
    parser.add_argument('--robust_start_epoch', type=int, default=1, help='Use schedule sampling')
    parser.add_argument('--robust_start', type=int, default=1, help='Use schedule sampling')
    parser.add_argument('--robust_end', type=float, default=0.9, help='Use schedule sampling')
    parser.add_argument('--robust_k', type=int, default=50, help='plot k/(k+exp(x/k)) from x=0 to 400, k=50')
    
    args = parser.parse_args()
    return args

