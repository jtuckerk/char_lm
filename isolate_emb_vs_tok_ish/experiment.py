import torch
import torch.nn as nn
from math import ceil
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import logging
import copy
import yaml
from collections import defaultdict
from torch.utils.checkpoint import checkpoint_sequential
import torch.nn.functional as F
import os

# from Huggingface transformers. split out to make more portable.
from modeling_distilbert import DistilBertForMaskedLM
import math

level = logging.getLevelName("INFO")
logging.basicConfig(
  level=level,
  format="[%(asctime)s] %(message)s",
  datefmt="%H:%M:%S",
)

Torch2Py = lambda x: x.cpu().numpy().tolist()

xent = nn.CrossEntropyLoss()

def xentropy_loss_fn(output, labels):
  return xent(output.view(-1, output.size(-1)), labels.view(-1))

def top1_acc_fn(output, labels):
  top = output.argmax(-1)
  right = top==labels
  return right.float().mean()

def Validate(val_loader, model, global_step, h, metric_tracker, val_type):
  cum_acc = 0
  cum_xent_loss = 0
  device = next(model.parameters()).device
  correct_words = []
  for i, data in enumerate(val_loader):
    data = {k: d.to(device) for k,d in data.items()}
    inputs = data['input_ids']
    labels = data['label_ids']

    with torch.no_grad():
      model.eval()
      outputs = model(inputs)
      xent_loss = xentropy_loss_fn(outputs, labels)
      cum_xent_loss += xent_loss      

      top1_acc = top1_acc_fn(outputs, labels)
      cum_acc += top1_acc

  steps = i+1
  metric_tracker.Track(val_type, 'global_step', global_step)
  metric_tracker.Track(val_type, 'accuracy', Torch2Py(cum_acc/steps))
  metric_tracker.Track(val_type, 'xent_loss', Torch2Py(cum_xent_loss.detach()/steps))
  
def Train(train_loader, model, loss_fn, optimizer, h, mt):

  device = next(model.parameters()).device
  n_batches = int(len(train_loader.dataset)/h['batch_size'])

  logging.info("TRAINING")
  tot_batches = n_batches*h['epochs']
  logging.info("total batches: %s" % tot_batches)
  global_step = 0

  acc_fn = top1_acc_fn
    
  bad_loss = False
  one_tenth = tot_batches // 10
  last_lr = optimizer.param_groups[0]['lr'] 
  for epoch in range(h['epochs']):
    start = time.time()
    running_loss = 0.0
    running_count = 0
    if bad_loss:
      break
    thrash_rate = 100
    for i, data in enumerate(train_loader):
      data = {k: d.to(device) for k,d in data.items()}
      global_step += 1
      inputs = data['input_ids']
      labels = data['label_ids']

      # zero the parameter gradients
      optimizer.zero_grad()

      outputs = model(inputs)
      
      loss = loss_fn(outputs, labels)

      if h.get('space_loss_weight', 0.0):
        switch_input = model.switch_input
        spaces = torch.where(inputs==1)
        act = torch.zeros_like(inputs).float()
        act[spaces[0], spaces[1]+1] = 1
        act[:, 0] = 1
        additional_loss = torch.nn.BCELoss()(switch_input, act)*h.get('space_loss_weight', 0.0)
        loss += additional_loss

      loss.backward()
      optimizer.step()

      # metrics
      running_loss += loss.item()
      running_count += 1

      if global_step % one_tenth == 0 or global_step < one_tenth and global_step % (one_tenth // 10) == 0 or global_step==1 or optimizer.param_groups[0]['lr'] != last_lr:
        ns_per_example = ((time.time()-start)/(h['batch_size']*(i+1)))*1e6
        accuracy = acc_fn(outputs, labels)
        mt.Track('train', 'accuracy', Torch2Py(accuracy))
          
        mt.Track('train', 'global_step', global_step)
        mt.Track('train', 'epoch', epoch)        
        mt.Track('train', 'loss', Torch2Py(loss.detach()), 3)
        if h.get('space_loss_weight', 0.0):
          mt.Track('train', 'space_loss', Torch2Py(additional_loss.detach()), 3)
        mt.Track('train', 'lr_at_step', optimizer.param_groups[0]['lr'])
        mt.Track('train', 'us/ex', ns_per_example)
        mt.Track('train', '% cmplt', 100*global_step/tot_batches, -2)
        mt.Log('train')
        
      if loss.item() > 1000 or torch.isnan(loss):
        bad_loss = True
        logging.info("Loss diverging. quitting. %s" % loss.item())
        break

      running_loss = running_count = 0.0
      last_lr = optimizer.param_groups[0]['lr']
  
  return not bad_loss, global_step

class CharToTokEmbDataset(torch.utils.data.Dataset):
  def __init__(self, torch_file, device='cuda'):
    self.data = torch.load(torch_file, map_location=device)
    
  def __len__(self):
    return len(self.data['token_ids'])

  def __getitem__(self, idx):
    item = {
      'input_ids': self.data['chars_encoded'][idx].long(),
      'label_ids': self.data['token_ids'][idx].long() 
      }
    return item

class ClozeDataset(torch.utils.data.Dataset):
  # expects torch file to be a single token sequence of a preprocessed text dataset.
  def __init__(self, torch_file, token_seq_length, percent_masks=.1, device='cuda'):
    self.mask_idx = 103
    # lowercased  mask
    self.percent_masks = percent_masks
    self.data = torch.load(torch_file, map_location=device)
    sequences = len(self.data)//token_seq_length
    self.data = self.data[:sequences*token_seq_length].reshape(sequences,-1)
    self.device=device

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    i = self.data[idx].long().to(self.device)
    masked = i.clone()
    mask = torch.rand(size=masked.size())
    mask = torch.where(mask<self.percent_masks)
    masked[mask] = self.mask_idx
    item = {
      'input_ids': masked,
      'label_ids': i
      }
    return item
  
# this is a weird dataset for inspection.
# we create a char encoded representation of each word padded with zeros
# its weird cuz the sequences are already tokenized and encoded but we're looking up the words and encoding them.
class ClozeDatasetWithCharInp(torch.utils.data.Dataset):
  # expects torch file to be a single token sequence of a preprocessed text dataset.
  def __init__(self, torch_file, token_seq_length, vocab_file,
               char_to_idx_file, word_length, percent_masks=.1, add_random_count=0, space_freq=1, add_next_words=0, device='cuda'):
    self.mask_idx = 103
    
    self.percent_masks = percent_masks
    self.add_random_count = add_random_count
    self.add_next_words = add_next_words
    self.space_freq = space_freq
    self.data = torch.load(torch_file, map_location=device)
    self.data = self.data.to(device)
    sequences = len(self.data)//token_seq_length
    self.data = self.data[:sequences*token_seq_length].reshape(sequences,-1)
    self.device=device

    self.word_char_encoding, self.char_to_idx_map = self._preprocess_chars(vocab_file, char_to_idx_file, word_length)
    self.word_char_encoding = self.word_char_encoding.to(device)

  def GetCharEncoding(self, word, char_to_idx_map, word_length):
    enc = [0]*word_length
    word = word.replace("##", "")
    for i, c in enumerate(word):
      if i>= word_length:
        break
      enc[i] = char_to_idx_map.get(c,0)
    return enc
    
  def _preprocess_chars(self, vocab_file, char_to_idx_file, word_length):
    bert_vocab = []
    self.suffix_words = set()
    with open(vocab_file) as f:
      for i, l in enumerate(f.readlines()):
        word = l.strip()
        if word.startswith("##"):
          word = word.replace("##", "")
          self.suffix_words.add(i)
        bert_vocab.append(word)

    longest_word_in_bert_vocab = max([len(w) for w in bert_vocab])

    char_to_idx_map = torch.load(char_to_idx_file)

    word_encodings = []
    for word in bert_vocab:
      encoding = self.GetCharEncoding(word.lower(), char_to_idx_map, word_length)
      word_encodings.append(encoding)

    word_char_encoding = torch.tensor(word_encodings, dtype=torch.int64)

    return word_char_encoding, char_to_idx_map
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    i = self.data[idx].long()
    masked = i.clone()
    rand = torch.rand(size=masked.size())
    mask = torch.where(rand<self.percent_masks)
    masked[mask] = self.mask_idx
    print(mask)
    item = {
      'label_ids': i,
      'token_ids': masked
      }
    char_encoded_input = self.word_char_encoding[masked]
    end = self.end_of_word(char_encoded_input)
    item['end_of_word_index'] = end
    if torch.rand(size=())<.9: # skip 1/10th of the time
      for i in range(self.add_random_count):
        char_encoded_input =self.add_random_next_word(char_encoded_input, end)
        end = self.end_of_word(char_encoded_input)

    for i in range(self.add_next_words):
      char_encoded_input = self.add_actual_next_word(char_encoded_input, end, masked, i)
      end = self.end_of_word(char_encoded_input)
      
    item['input_ids'] = char_encoded_input

    return item
  def end_of_word(self, char_encoded_input):
    chars = char_encoded_input!=0
    return chars.sum(-1)

  def IsSuffix(self, word_idx):
    if word_idx==0:
      return False
    return word_idx.item() in self.suffix_words
    
  def append_single(self, char_encoded_input, add_word, end):
    c_len = len(char_encoded_input)
    if torch.rand(size=())>=self.space_freq and self.IsSuffix(add_word):
      end = torch.clamp(end, max=c_len-1)
      char_encoded_input[end:] = self.word_char_encoding[add_word][:c_len-end]
    else:
      char_encoded_input[end:end+1] = self.char_to_idx_map[' '] # space char
      end = torch.clamp(end, max=c_len-1)
      char_encoded_input[end+1:] = self.word_char_encoding[add_word][:c_len-end-1]
    return char_encoded_input
      
  def add_random_next_word(self, char_encoded_input, end):
    # returns the word with a random word added at the end of the word
    char_encoded_input = char_encoded_input.clone()
    if len(char_encoded_input.size())>=2:
      add_word = torch.randint(0, len(self.word_char_encoding), (len(char_encoded_input),))
      new_words=[]
      for i, single_encoded in enumerate(char_encoded_input):
        new_words.append(self.append_single(single_encoded, add_word[i], end[i]))
      char_encoded_input = torch.stack(new_words)
    else:
      add_word = torch.randint(0, len(self.word_char_encoding), ())
      char_encoded_input = self.append_single(char_encoded_input, add_word, end)

    return char_encoded_input

  def add_actual_next_word(self, char_encoded_input, end, word_idxs, count):
    # returns the word with a random word added at the end of the word
    char_encoded_input = char_encoded_input.clone()
    new_words=[]

    for i, single_encoded in enumerate(char_encoded_input):
      pos = i+count+1

      if pos >= len(word_idxs):
        next_word = 0 # TODO fragile
      else:
        next_word = word_idxs[pos]
        
      new_words.append(self.append_single(single_encoded, next_word, end[i]))
    char_encoded_input = torch.stack(new_words)

    return char_encoded_input
  

class LambdaLayer(nn.Module):
  def __init__(self, lambd):
    super(LambdaLayer, self).__init__()
    self.lambd = lambd
  def forward(self, x):
    return self.lambd(x)

class HardLeakyGradSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(self,x):
        self.leaky = (x < 0) | (x > 1)
        return x.clamp(min=0.0, max=1.0)
    @staticmethod
    def backward(self,grad_output):
        grad_input = grad_output.clone()
        grad_input[self.leaky] *= 0.1
        
        return grad_input

def hard_sigmoid(x):
  return x.clamp(min=0.0, max=1.0)

def hard_leaky_grad_sigmoid(x):
    return HardLeakyGradSigmoid.apply(x)


class LSTMSwitchboard(nn.Module):
  def __init__(self, out_size, layers, sigmoid_out=False):
    super(LSTMSwitchboard,  self).__init__()
    self.sigmoid_out = sigmoid_out
    self.rnn = torch.nn.LSTM(1, out_size, layers, batch_first=True)
    self.expand_weight = nn.Linear(out_size, out_size)
    self.sigmoid = torch.nn.Sigmoid()
    
  def forward(self, x):
    # takes input of batchsizeXseqlengthx1
    x = x.unsqueeze(-1)
    out, h = self.rnn(x)
    out = self.expand_weight(out)
    if self.sigmoid_out == 'standard':
      out = self.sigmoid(out)
    elif self.sigmoid_out == 'hard':
      out = hard_sigmoid(out)
    elif self.sigmoid_out == 'hard_leaky':
      out = hard_leaky_grad_sigmoid(out)
    return out.permute(0,2,1)
  

class GRUSwitchboard(nn.Module):
  def __init__(self, out_size, layers, sigmoid_out=False):
    super(GRUSwitchboard,  self).__init__()
    self.sigmoid_out = sigmoid_out 
    self.rnn = torch.nn.GRU(1, out_size, layers, batch_first=True)
    # I thnik this will make it easier to the model to force either 0s or 1s. 
    self.sigmoid = torch.nn.Sigmoid()
    
  def forward(self, x):
    # takes input of batchsizeXseqlengthx1
    x = x.unsqueeze(-1)
    out, h = self.rnn(x)
    if self.sigmoid_out:
      out = self.sigmoid(out)
    return out

class SwitchboardAttention(nn.Module):
  # Differentiable state machine based attention to select items in a long sequence to be kept.
  def __init__(self, out_size, sigmoid=False):
    super(SwitchboardAttention, self).__init__()
    self.out_size = out_size
    self.sigmoid = sigmoid
    
    n_range = torch.arange(0,out_size).long()

    # Set up state shift mechanism
    self.translation = torch.zeros(size=(out_size,out_size))
    self.translation[n_range[:out_size-1], n_range[:out_size-1]+1] =1
    self.translation.requires_grad = False
    self.identity = torch.zeros(size=(out_size,out_size))
    self.identity[n_range, n_range] = 1
    self.identity.requires_grad = False

    self.states = None

  def _apply(self, fn):
    super(SwitchboardAttention, self)._apply(fn)
    self.translation = fn(self.translation)
    self.identity = fn(self.identity)
    
  def _reset_states(self, x):
    batch_size = x.shape[0]
    self.states = torch.zeros(size=(batch_size, self.out_size), requires_grad=False, device=x.device)
    
    self.states[:, 0] = 1
    
  def forward(self, x):
    self._reset_states(x)

    flatten = x
    if self.sigmoid == 'standard':
      flatten = torch.nn.Sigmoid()(x)
    elif self.sigmoid == 'hard':
      flatten = hard_sigmoid(x)
    elif self.sigmoid == 'hard_leaky':
      flatten = hard_leaky_grad_sigmoid(x)
    self.switch_input = flatten
    states_list = []
    for i in flatten.T:
      update = self.states*i.unsqueeze(-1)
      i_inv = 1-i
      keep = self.states*i_inv.unsqueeze(-1)
      states_list.append(update)
      self.states = update@self.translation + keep@self.identity
      
    state = torch.stack(states_list)
    return state.permute(1,2,0)

class PadLayer(nn.Module):
  def __init__(self, pad_size, start_pad=0):
    super(PadLayer, self).__init__()
    self.pad_size = pad_size
    self.start_pad = start_pad

  def forward(self, x):
    return F.pad(x, (self.start_pad, self.pad_size-self.start_pad))      

class ConvSegment(nn.Module):
  def __init__(self, input_channels, activation, kernel_filters_sizes, center_start_pad=False):
    super(ConvSegment, self).__init__()
    if activation == 'relu':
      self.conv_activation = nn.ReLU()
    if activation == 'gelu':
      Gelu = LambdaLayer(gelu)
      self.conv_activation = Gelu
    elif activation == 'sigmoid':
      self.conv_activation = nn.Sigmoid()

    self.convs = []

    first_conv=True
    last_out_chan = input_channels
    for k, f in kernel_filters_sizes:
      start_pad = k//2 if first_conv and center_start_pad else 0
      self.convs.append(PadLayer(k-1, start_pad))
      self.convs.append(nn.Conv1d(last_out_chan, f, k))
      self.convs.append(self.conv_activation)
      last_out_chan = f
      first_conv = False
      
    self.convs = nn.Sequential(*self.convs)

  def forward(self, x):
    return self.convs(x)
  
class IdentityConv(nn.Module):
  # applies convolution with an identity kernel.
  # in other words: unfolds(repeats with a stride of 1) the input in kernel_size chunks.
  def __init__(self, kernel_size):
    super(IdentityConv, self).__init__()
    self.kernel_size = kernel_size
  def forward(self, x):
    x = x.permute(0,2,1)
    unfolded = x.unfold(1,self.kernel_size,1)
    #reshape as (N,C,W) (batch size, channels, width)
    # each channel has the kern_size_charsXembedding unfolded to a single dimension. 12 chars & 4 dim emb = C of 48
    return  unfolded.flatten(-2).permute(0,2,1)
  
class TokensToEmb(nn.Module):
  def __init__(self, h):    
    super(TokensToEmb, self).__init__()
    self.emb = nn.Embedding(h['char_vocab_size'], h['char_embedding_size'])

    # self.segment_attn = ConvSegment(h['char_embedding_size'], h['conv_activation'], h['seg_attn.kernel|filter_sizes'])
    # seg_attn_out_size = h['seg_attn.kernel|filter_sizes'][-1][1]

    # seg1 takes in the attention output and the char embeddings
    if h['seg1_type']=='conv':
      self.segment1 = ConvSegment(h['char_embedding_size'], h['conv_activation'], h['seg1.kernel|filter_sizes'])
      seg1_out_size = h['seg1.kernel|filter_sizes'][-1][1]
    elif h['seg1_type']=='unfold':
      self.segment1 = IdentityConv(h['seg1.kernel_size'])
      seg1_out_size = h['seg1.kernel_size']*h['char_embedding_size']
      
    self.segment2 = ConvSegment(seg1_out_size, h['conv_activation'], h['seg2.kernel|filter_sizes'])
    seg2_out_size = h['seg2.kernel|filter_sizes'][-1][1]
    self.final_conv = nn.Conv1d(seg2_out_size, h['token_embedding_size'], 1)

    # this probs isnt necessary - but if i decide to change the shapes it might be useful later.
    self.end_of_word_weight = h.get('end_of_word_loss_weight', 0)
    if self.end_of_word_weight:
      self.end_word_dense = torch.nn.Linear(h['word_length'], h['word_length'])
      self.end_word_softmax = torch.nn.Softmax(-1)

  def forward_first_half(self, x):
    embeddings  = self.emb(x).permute(0,2,1)

    seg1_out = self.segment1(embeddings)
    return seg1_out

  def forward_second_half(self, x):
    seg2_out = self.segment2(x)
    embedding_out = self.final_conv(seg2_out).permute(0,2,1)
    return embedding_out
  
  def forward(self, x):
    # Only generate 1 embedding no matter what the first few layers of conv produce
    h1 = self.forward_first_half(x)
    self.h1 = h1
    embedding_out = self.forward_second_half(h1)
    return embedding_out.squeeze(1)

class TokStartAttn(nn.Module):
  def __init__(self, h):
    super(TokStartAttn, self).__init__()
    self.emb = nn.Embedding(h['char_vocab_size'], h['char_embedding_size'])

    self.char_input_window_size = h['attn_conv.kernel|filter_sizes'][0][0]
    self.conv = ConvSegment(h['char_embedding_size'], h['conv_activation'], h['attn_conv.kernel|filter_sizes'], True)
    conv_out_size = h['attn_conv.kernel|filter_sizes'][-1][1]

    # no nonlinearity
    self.final_conv = nn.Conv1d(conv_out_size, 1, 1)

  def forward(self, x):

    embout = self.emb(x).permute(0,2,1)
    char_block = self.conv(embout)
    char_block_logits = self.final_conv(char_block).permute(0,2,1).squeeze(-1)

    return char_block_logits

# copy paste cuz its easier. TODO clean up if i decide to do lots of experiments and their results matter
class CharEmbedderHalf(nn.Module):
  def __init__(self, h, device='cuda'):    
    super(CharEmbedderHalf, self).__init__()
    # can it be input sequence size agnostic? I think so. just trim based on input size dynamically.
    self.out_seq_size = h['token_seq_length']

    self.tokens_to_emb = TokensToEmb(h)

    self.char_input_window_size = h['seg1.kernel_size']

    self.manual_attn = h.get('manual_attention', False)
  
  def forward(self, x):
    # input character IDs as longs in shape batchXseq_lenXword_len

    # sb shape: batchXOut_seq_sizeXin_seq_size
    # char blocks shape batchXseqXembedding_size
    dense_char_blocks = self.tokens_to_emb.emb(x).permute(0,1,3,2).flatten(-2)
    self.dense_char_blocks = dense_char_blocks
    # convert the char blocks into token embeddings
    out_embeddings = self.tokens_to_emb.forward_second_half(dense_char_blocks.permute(0,2,1))

    return out_embeddings
  
class CharEmbedder(nn.Module):
  def __init__(self, h, device='cuda'):    
    super(CharEmbedder, self).__init__()
    # can it be input sequence size agnostic? I think so. just trim based on input size dynamically.
    self.out_seq_size = h['token_seq_length']

    self.tokens_to_emb = TokensToEmb(h)

    self.char_input_window_size = h['seg1.kernel_size']

    self.manual_attn = h.get('manual_attention', False)
    if not self.manual_attn:
      self.tokens_start_attn = TokStartAttn(h)
    # dense layers project the attention layer to an attention matrix or switchboard
    if h['switchboard_type'] == 'lstm':
      self.sb_module = LSTMSwitchboard(h['token_seq_length'], h['rnn_hidden_layers'], h['sigmoid_out'])
    if h['switchboard_type'] == 'gru':
      self.sb_module = GRUSwitchboard(h['token_seq_length'], h['rnn_hidden_layers'], h['sigmoid_out'])
    if h['switchboard_type'] == 'rule_based':
      self.sb_module = SwitchboardAttention(h['token_seq_length'], h['sb.sigmoid'])
    
  def GetSwitchboard(self, attn):
    out = self.sb_module(attn)
    #self.switch_input = self.sb_module.switch_input
    return out
  
  def forward(self, x):
    # input character IDs as longs

    if self.manual_attn:
      spaces = torch.where(x==1)
      act = torch.zeros_like(x)

      # TODO clean and include other special chars.
      act[spaces[0], torch.clamp(spaces[1]+1, max=act.shape[-1]-1)] = 1

      act[torch.where(act==0)]=-1
      act[:, 0] = 1
      act*=100
      char_block_attn = act.float()
    else:
      char_block_attn = self.tokens_start_attn(x)

    x = F.pad(x, (0, self.char_input_window_size-1))
    # Get blocks of embedded chars
    char_blocks = self.tokens_to_emb.forward_first_half(x).permute(0,2,1)
    self.char_blocks = char_blocks

    self.char_block_attn = char_block_attn

    # Use the switchboard (like SortCut) to get only some the char blocks.
    self.switchboard = self.GetSwitchboard(char_block_attn)
    # sb shape: batchXOut_seq_sizeXin_seq_size
    # char blocks shape batchXseqXembedding_size
    dense_char_blocks = self.switchboard@char_blocks
    self.dense_char_blocks = dense_char_blocks
    # convert the char blocks into token embeddings
    out_embeddings = self.tokens_to_emb.forward_second_half(dense_char_blocks.permute(0,2,1))

    return out_embeddings

def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
  
class Net(nn.Module):
  def __init__(self, h, device='cuda'):    
    super(Net, self).__init__()
    # takes in token ids or word embeddings.
    self.bert = DistilBertForMaskedLM.from_pretrained(h['bert_checkpoint'])
    self.input_type = h['input_type']
    if h['input_type'] == 'token_encoded':
      self.token_encoded = True
    elif h['input_type'] == 'char':
      self.position_embeddings = h.get('position_embeddings', False)

      
      self.token_encoded = False
      self.char_embedder = CharEmbedder(h)
      self.position_embeddings = self.bert.distilbert.embeddings.position_embeddings
      self.LayerNorm = self.bert.distilbert.embeddings.LayerNorm
      self.dropout = nn.Dropout(.1)

    elif h['input_type'] == 'tokenized_chars':
      self.position_embeddings = h.get('position_embeddings', False)

      self.token_encoded = False
      self.char_embedder = CharEmbedderHalf(h)
      self.position_embeddings = self.bert.distilbert.embeddings.position_embeddings
      self.LayerNorm = self.bert.distilbert.embeddings.LayerNorm
      self.dropout = nn.Dropout(.1)
      
    self.softmax = torch.nn.Softmax(-1)
    self.skip_bert = h.get('skip_bert', False)

  def forward(self, x):
    if self.input_type == 'token_encoded':
      word_logits = self.bert(input_ids=x)[0]
    elif 'char' in self.input_type:
      embedded_chars = self.char_embedder(x)
      self.embedded_chars_no_pos = embedded_chars.clone()
      #self.switch_input = self.char_embedder.switch_input
      if self.skip_bert:
        word_logits = embedded_chars@self.bert.distilbert.embeddings.word_embeddings.weight.data.T  # (bs, seq_length, vocab_size)
      else:
        self.bert.distilbert.inputs_embeds = None
        self.bert.distilbert.inputs_embeds_no_pos = None
        if self.position_embeddings:
          seq_length = embedded_chars.size(1)
          position_ids = torch.arange(seq_length, dtype=torch.long, device=embedded_chars.device)  # (max_seq_length)
          self.position_ids = position_ids.unsqueeze(0).expand(embedded_chars.size()[:2])  # (bs, max_seq_length)
          self.pos_emb_inst = self.position_embeddings(self.position_ids)
          embedded_chars += self.pos_emb_inst
          self.embedded_chars = embedded_chars
          
        embedded_chars = self.LayerNorm(embedded_chars)  # (bs, max_seq_length, dim)
        embedded_chars = self.dropout(embedded_chars)  # (bs, max_seq_l
        word_logits = self.bert(inputs_embeds=embedded_chars)[0]

    return word_logits
  
def InitDataset(exp_info, dev='cuda'):
  if exp_info['input_type'] == 'token_encoded':
    ds =  ClozeDataset(exp_info['dataset'], exp_info['token_seq_length'], exp_info['percent_masks'], device=dev)

  elif exp_info['input_type'] == 'char':
    ds = CharToTokEmbDataset(exp_info['dataset'], device=dev)
  elif exp_info['input_type'] == 'tokenized_chars':
    ds = ClozeDatasetWithCharInp(exp_info['dataset'],
                                 exp_info['token_seq_length'],
                                 exp_info['vocab_file'],
                                 exp_info['char_to_idx_file'],
                                 exp_info['word_length'],
                                 exp_info['percent_masks'],
                                 exp_info['add_random_count'],
                                 exp_info['space_frequency'],
                                 exp_info['add_next_words'],
                                 device='cpu')

  tr, va = exp_info['dataset_split'] # the rest is test
  n_train_examples, n_val_examples = int(len(ds)*tr), int(len(ds)*va)
  n_test_examples = len(ds)- n_train_examples - n_val_examples
  train_d, val_d, test_d = torch.utils.data.random_split(
    ds, [n_train_examples,n_val_examples,n_test_examples])
  logging.debug("n_train_examples: %s" % n_train_examples)
  logging.debug("n_val_examples: %s" % n_val_examples)

  return train_d, val_d, test_d

def CreateModel(h):
  if 'char' in h['input_type'] and 'seg2.kernel|filter_sizes' not in h:
    # find the emb_pred model shape from the emb_pred exp
    exphash = h['model_checkpoint'].split('/')[-1]
    fname = 'emb_pred_results/' + exphash
    with open(fname) as f:
      results = yaml.safe_load(f.read())
      sizes = results['exp_info']['hyperparameters']['seg2.kernel|filter_sizes']
      h['seg2.kernel|filter_sizes'] = sizes
  model = Net(h)
  return model

def LoadCheckpoint(model, checkpoint_file):
  r = model.load_state_dict(torch.load(checkpoint_file), strict=False)
  logging.info("Loaded model: %s" % str(r))

def SaveModel(model, filename):
  torch.save(model.state_dict(), filename)

def CleanupExperiment(model):
  del model
  torch.cuda.empty_cache()

def RunOne(h, model, data, mt, dev='cuda'):
  train_d, val_d, test_d = data
  model.to(dev)

  if 'seed' in h:
    torch.manual_seed(h['seed'])

  train_loader = DataLoader(train_d, batch_size=h['batch_size'], shuffle=True, num_workers=os.cpu_count())
  validation_loader = DataLoader(val_d, batch_size=h['batch_size'], shuffle=False, num_workers=os.cpu_count())
  logging.debug("n_train_batches: %s" % (len(train_loader.dataset)//h['batch_size']))

  char_str = ""
  if 'char' in h['input_type']:
    char_str = str(model.char_embedder)
  logging.info("Experiment Model:\n" + "(bert)\n"+char_str)
  logging.info("Hyperparams:\n" + str(h))

  size_params = 0
  size_bytes = 0

  for p in model.parameters():
    size_params += p.nelement()
    size_bytes += p.nelement()*p.element_size()
  logging.info("Model Size: {:.2e} bytes".format(size_bytes))

  if 'model_size_range_bytes' in h:
    min_size, max_size = h['model_size_range_bytes']
    if size_bytes < min_size or size_bytes > max_size:
      logging.info("Model size (%s bytes) outside of acceptable range. skipping" % size_bytes)
      return {'val': {'size_bytes': size_bytes,
                      'size_params': size_params},
              'exit_info': 'size outside of range. skipping'}, model

  if 'bert' in h['freeze_modules']:
    for param in model.bert.parameters():
      param.requires_grad = False
      
  if 'embedding_predict' in h['freeze_modules']:
    for param in model.char_embedder.tokens_to_emb.parameters():
      param.requires_grad = False

  if 'attn_conv' in h['freeze_modules']:      
    for param in model.char_embedder.tokens_start_attn.parameters():
      param.requires_grad = False      

  if 'sb_module' in h['freeze_modules']:      
    for param in model.sb_module.parameters():
      param.requires_grad = False

  logging.info("Params req_grad:")
  for name, param in model.named_parameters():
    if 'bert' not in name:
      logging.info("%s: %s" %(name, param.requires_grad))
    
  if h['optimizer'] == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=h['learning_rate'])
  if h['loss_fn'] == 'xentropy':
    loss_fn = xentropy_loss_fn

  success, step = Train(train_loader,
                        model, loss_fn, optimizer, h, mt)

  if success and h['run_validation']:
    Validate(validation_loader, model, step, h, mt, 'val')
    mt.Log('val')

  mt.AddExpInfo('size_params', size_params)
  mt.AddExpInfo('size_bytes', size_bytes)
  if dev=='cuda':
    mt.AddExpInfo('max_mem_alloc', torch.cuda.max_memory_allocated())
    torch.torch.cuda.reset_max_memory_allocated()
  return model
