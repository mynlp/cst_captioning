import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

import pdb

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()
    
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        
        output = - input.gather(1, target) * mask
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
    
    def forward(self, feats):
        """
        feats is a list, each element is a tensor that have size (N x C x F)
        at the moment assuming that C == 1
        """
        out = torch.cat([m(feats[i].squeeze(1)) for i,m in enumerate(self.feat_list)])
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
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        
        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size, 
                self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

    def forward(self, xt, state):
        output, state = self.rnn(xt.unsqueeze(0), state)
        return output.squeeze(0), state
    
    
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
        self.seq_per_img = vars(opt).get('seq_per_img', 1)
        
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()
        self.feat_pool = FeatPool(self.feat_dims, self.num_layers * self.rnn_size, self.drop_prob_lm)
        self.feat_expander = FeatExpander(self.seq_per_img)
        
        opt.feat_size = len(self.feat_dims)
        self.core = RNNUnit(opt)

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

        for i in range(seq.size(1)):
            if i == 0:
                xt = fc_feats
            else:    
                it = seq[:, i-1].clone()
                # break if all the sequences end
                if i >= 2 and seq[:, i-1].data.sum() == 0:
                    break
                xt = self.embed(it)
                
            output, state = self.core(xt, state)
            output = F.log_softmax(self.logit(self.dropout(output)))
            outputs.append(output)

        # output size is: B x L x V
        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1)

    def sample(self, feats, opt={}):
        beam_size = opt.get('beam_size', 1)
        if beam_size > 1:
            return self.sample_beam(feats, opt)

        fc_feats = self.feat_pool(feats)
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = fc_feats
            else:
                if t == 1: # input <bos>
                    it = fc_feats.data.new(batch_size).long().zero_()
                else:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
                    
                xt = self.embed(Variable(it, requires_grad=False))

            if t >= 2:
                if t == 2:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core(xt, state)
            logprobs = F.log_softmax(self.logit(output))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
    
    def sample_beam(self, feats, opt={}):
        beam_size = opt.get('beam_size', 5)
        fc_feats = self.feat_pool(feats)
        batch_size = fc_feats.size(0)
        
        assert beam_size <= self.vocab_size + 1
        
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam
            for t in range(self.seq_length + 2):
                if t == 0:
                    xt = fc_feats[k].expand(beam_size, self.input_encoding_size)
                elif t == 1: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
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
                    if t == 2:  # at first time step only the first beam is active
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
                    if t > 2:
                        # well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:t-2].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-2].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if t > 2:
                            beam_seq[:t-2, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t-2, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at vix
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']] # dimension one is time step

                        # append new end terminal at the end of this beam
                        beam_seq[t-2, vix] = v['c'] # c'th word is the continuation
                        beam_seq_logprobs[t-2, vix] = v['r'] # the raw logprob here
                        beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam

                        if v['c'] == 0 or t == self.seq_length + 1:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(), 
                                                'logps': beam_seq_logprobs[:, vix].clone(),
                                                'p': beam_logprobs_sum[vix]
                                                })
        
                    # encode as vectors
                    it = beam_seq[t-2]
                    xt = self.embed(Variable(it.cuda()))
                
                if t >= 2:
                    state = new_state

                output, state = self.core(xt, state)
                logprobs = F.log_softmax(self.logit(output))

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)
