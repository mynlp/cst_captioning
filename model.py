import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, seq, sample_logprobs, reward):
        
        # import pdb; pdb.set_trace()
        
        sample_logprobs = to_contiguous(sample_logprobs).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        # add one to the right to count for the <eos> token
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - sample_logprobs * reward * Variable(mask)
        output = torch.sum(output) / torch.sum(mask)

        return output
    
class CrossEntropyCriterion(nn.Module):
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def forward(self, pred, target, mask):
        # truncate to the same size
        target = target[:, :pred.size(1)]
        mask =  mask[:, :pred.size(1)]
        
        pred = to_contiguous(pred).view(-1, pred.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        
        output = -pred.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
    
class FeatPool(nn.Module):
    def __init__(self, feat_dims, out_size, dropout):
        super(FeatPool, self).__init__()
        
        module_list = []
        for dim in feat_dims:
            module = nn.Sequential(nn.Linear(dim, out_size), nn.ReLU(), nn.Dropout(dropout))
            module_list += [module]
        self.feat_list = nn.ModuleList(module_list)
        
        # self.embed = nn.Sequential(nn.Linear(sum(feat_dims), out_size), nn.ReLU(), nn.Dropout(dropout))
    
    def forward(self, feats):
        """
        feats is a list, each element is a tensor that have size (N x C x F)
        at the moment assuming that C == 1
        """
        out = torch.cat([m(feats[i].squeeze(1)) for i,m in enumerate(self.feat_list)], 1)
        # pdb.set_trace()
        # out = self.embed(torch.cat(feats, 2).squeeze(1))
        return out

class FeatExpander(nn.Module):
    def __init__(self, n=1):
        super(FeatExpander, self).__init__()
        self.n = n
    
    def forward(self, x):
        if self.n == 1:
            out = x
        else:
            out = Variable(x.data.new(self.n*x.size(0), x.size(1)), volatile=x.volatile)
            for i in range(x.size(0)):
                out[i*self.n:(i+1)*self.n] = x[i].expand(self.n, x.size(1))
        return out
    
    def set_n(self, x):
        self.n = x
        
class RNNUnit(nn.Module):
    def __init__(self, opt):
        super(RNNUnit, self).__init__()
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        
        if opt.model_type == 'standard':
            self.input_size = opt.input_encoding_size
        elif opt.model_type in ['concat', 'manet']:
            self.input_size = opt.input_encoding_size + opt.video_encoding_size
            
        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_size, 
                self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

    def forward(self, xt, state):
        output, state = self.rnn(xt.unsqueeze(0), state)
        return output.squeeze(0), state
    

class MANet(nn.Module):
    """
    MANet: Modal Attention 
    """
    def __init__(self, video_encoding_size, rnn_size, num_feats):
        super(MANet, self).__init__()
        self.video_encoding_size = video_encoding_size
        self.rnn_size = rnn_size
        self.num_feats = num_feats
        
        self.f_feat_m = nn.Linear(self.video_encoding_size, self.num_feats)
        self.f_h_m = nn.Linear(self.rnn_size, self.num_feats)
        self.align_m = nn.Linear(self.num_feats, self.num_feats)
        
    def forward(self, x, h):
        f_feat = self.f_feat_m(x)
        f_h = self.f_h_m(h.squeeze(0)) # assuming now num_layers is 1
        att_weight = nn.Softmax()(self.align_m(nn.Tanh()(f_feat + f_h)))
        att_weight = att_weight.unsqueeze(2).expand(x.size(0), self.num_feats, self.video_encoding_size/self.num_feats)
        att_weight = att_weight.contiguous().view(x.size(0), x.size(1))
        return x*att_weight
    
class CaptionModel(nn.Module):
    """
    A baseline captioning model
    """
    def __init__(self, opt):
        super(CaptionModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.feat_dims = opt.feat_dims
        self.num_feats = len(self.feat_dims)
        self.seq_per_img = opt.train_seq_per_img
        self.model_type = opt.model_type
        self.bos_index = 1 # index of the <bos> token
        self.ss_prob = 0
        
        self.embed = nn.Embedding(self.vocab_size, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()
        self.feat_pool = FeatPool(self.feat_dims, self.num_layers * self.rnn_size, self.drop_prob_lm)
        self.feat_expander = FeatExpander(self.seq_per_img)
        
        self.video_encoding_size = self.num_feats * self.num_layers * self.rnn_size
        opt.video_encoding_size = self.video_encoding_size
        self.core = RNNUnit(opt)
        
        if self.model_type == 'manet':
            self.manet = MANet(self.video_encoding_size, self.rnn_size, self.num_feats)

    def set_ss_prob(self, p):
        self.ss_prob = p
        
    def set_seq_per_img(self, x):
        self.seq_per_img = x
        self.feat_expander.set_n(x)
        
    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
        if self.rnn_type == 'lstm':
            return (Variable(weight.new(self.num_layers, batch_size, self.rnn_size).zero_()),
                    Variable(weight.new(self.num_layers, batch_size, self.rnn_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers, batch_size, self.rnn_size).zero_())

    def forward(self, feats, seq):

        fc_feats = self.feat_pool(feats)
        fc_feats = self.feat_expander(fc_feats)
        
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []

        # -- if <image feature> is input at the first step, use index -1
        # -- the <eos> token is not used for training
        start_i = -1 if self.model_type == 'standard' else 0
        end_i = seq.size(1) - 1
        
        for token_idx in range(start_i, end_i):
            if token_idx == -1:
                xt = fc_feats
            else:   
                # token_idx = 0 corresponding to the <BOS> token 
                # (already encoded in seq)
                
                if self.training and token_idx >= 1 and self.ss_prob > 0.0:
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, token_idx].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, token_idx].data.clone()
                        prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                        sample_ind_tokens = torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind)
                        it.index_copy_(0, sample_ind, sample_ind_tokens)
                        it = Variable(it, requires_grad=False)
                else:
                    it = seq[:, token_idx].clone()
                
                # break if all the sequences end, which requires EOS token = 0
                if it.data.sum() == 0:
                    break
                xt = self.embed(it)
                
            if self.model_type == 'standard':
                output, state = self.core(xt, state)
            else:    
                if self.model_type == 'manet':
                    fc_feats = self.manet(fc_feats, state[0])
                output, state = self.core(torch.cat([xt, fc_feats], 1), state)
                
            if token_idx >= 0:
                output = F.log_softmax(self.logit(self.dropout(output)))
                outputs.append(output)
            
        # only returns outputs of seq input
        # output size is: B x L x V (where L is truncated lengths
        # which are different for different batch)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def sample(self, feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        expand_feat= opt.get('expand_feat', 0)
        
        if beam_size > 1:
            return self.sample_beam(feats, opt)

        fc_feats = self.feat_pool(feats)
        if expand_feat == 1:  
            fc_feats = self.feat_expander(fc_feats)
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        seq = []
        seqLogprobs = []
        
        unfinished = fc_feats.data.new(batch_size).fill_(1).byte()
        
        # -- if <image feature> is input at the first step, use index -1
        start_i = -1 if self.model_type == 'standard' else 0
        end_i = self.seq_length - 1
        
        for token_idx in range(start_i, end_i):
            if token_idx == -1:
                xt = fc_feats
            else:
                if token_idx == 0: # input <bos>
                    it = fc_feats.data.new(batch_size).long().fill_(self.bos_index)
                elif sample_max == 1:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                    it = torch.multinomial(prob_prev, 1).cuda()
                    # gather the logprobs at sampled positions
                    sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) 
                    # and flatten indices for downstream processing
                    it = it.view(-1).long() 
                    
                xt = self.embed(Variable(it, requires_grad=False))

            if token_idx >= 1:
                unfinished = unfinished * (it > 0)
                
                # 
                it = it * unfinished.type_as(it)
                seq.append(it) 
                seqLogprobs.append(sampleLogprobs.view(-1))
                
                # requires EOS token = 0
                if unfinished.sum() == 0:
                    break

            if self.model_type == 'standard':
                output, state = self.core(xt, state)
            else:    
                if self.model_type == 'manet':
                    fc_feats = self.manet(fc_feats, state[0])
                output, state = self.core(torch.cat([xt, fc_feats], 1), state)
                
            logprobs = F.log_softmax(self.logit(output))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
    
    def sample_beam(self, feats, opt={}):
        beam_size = opt.get('beam_size', 5)
        fc_feats = self.feat_pool(feats)
        batch_size = fc_feats.size(0)
        
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            fc_feats_k = fc_feats[k].expand(beam_size, self.video_encoding_size)
            
            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam
            
            # -- if <image feature> is input at the first step, use index -1
            start_i = -1 if self.model_type == 'standard' else 0
            end_i = self.seq_length - 1
            
            for token_idx in range(start_i, end_i):
                if token_idx == -1:
                    xt = fc_feats_k
                elif token_idx == 0: # input <bos>
                    it = fc_feats.data.new(beam_size).long().fill_(self.bos_index)
                    xt = self.embed(Variable(it, requires_grad=False))
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float() # lets go to CPU for more efficiency in indexing operations
                    ys,ix = torch.sort(logprobsf,1,True) # sorted array of logprobs along each previous beam (last true = descending)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if token_idx == 1:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in (sorted) position c
                            local_logprob = ys[q,c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append({'c':ix.data[q,c], 'q':q, 'p':candidate_logprob.data[0], 'r':local_logprob.data[0]})
                    candidates = sorted(candidates, key=lambda x: -x['p'])
                    
                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if token_idx > 1:
                        # well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:token_idx-1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:token_idx-1].clone()
                        
                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if token_idx > 1:
                            beam_seq[:token_idx-1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:token_idx-1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at vix
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']] # dimension one is time step

                        # append new end terminal at the end of this beam
                        beam_seq[token_idx-1, vix] = v['c'] # c'th word is the continuation
                        beam_seq_logprobs[token_idx-1, vix] = v['r'] # the raw logprob here
                        beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam

                        if v['c'] == 0 or token_idx == self.seq_length-2:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(), 
                                                'logps': beam_seq_logprobs[:, vix].clone(),
                                                'p': beam_logprobs_sum[vix],
                                                'ppl': np.exp(-beam_logprobs_sum[vix]/(token_idx-1))
                                                })
        
                    # encode as vectors
                    it = beam_seq[token_idx-1]
                    xt = self.embed(Variable(it.cuda()))
                
                if token_idx >= 1:
                    state = new_state

                if self.model_type == 'standard':
                    output, state = self.core(xt, state)
                else:    
                    if self.model_type == 'manet':
                        fc_feats_k = self.manet(fc_feats_k, state[0])
                    output, state = self.core(torch.cat([xt, fc_feats_k], 1), state)

                logprobs = F.log_softmax(self.logit(output))

            #self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: x['ppl'])
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
            
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

