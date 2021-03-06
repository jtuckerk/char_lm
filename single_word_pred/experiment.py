# experiment to predict a word(onehot encoding) from characters
# copied from single_emb_pred, except predicts onehot instead of word embedding.
# the dataset can be configured to add words after the word of interest and generate misspellings.

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
import os

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

def acc_fn(output, labels):
  top = output.argmax(-1)
  right = top==labels
  return right.float().mean()

def Validate(val_loader, model, global_step, h, metric_tracker, val_type):
  device = next(model.parameters()).device

  cum_acc = 0
  cum_xent_loss = 0

  correct_words = []
  for i, data in enumerate(val_loader):
    data = {k: d.to(device) for k,d in data.items()}
    inputs = data['features'].to(device)
    labels = data['labels'].long()

    with torch.no_grad():
      outputs = model(inputs)
      xent_loss = xentropy_loss_fn(outputs, labels)
      cum_xent_loss += xent_loss

      if h.get('eval_acc', False):
        acc = acc_fn(outputs, labels)
        cum_acc += acc

  steps = i+1
  metric_tracker.Track(val_type, 'global_step', global_step)
  if h.get('eval_acc', False):
    metric_tracker.Track(val_type, 'accuracy', Torch2Py(cum_acc/steps))
  metric_tracker.Track(val_type, 'xent_loss', Torch2Py(cum_xent_loss.detach()/steps))

def Train(train_loader, model, loss_fn, optimizer, h, mt):

  device = next(model.parameters()).device

  n_batches = int(len(train_loader.dataset)/h['batch_size'])

  logging.info("TRAINING")
  tot_batches = n_batches*h['epochs']
  logging.info("total batches: %s" % tot_batches)
  global_step = 0

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=h['lr_step_size'], gamma=h['lr_decay'])
  bad_loss = False
  one_tenth = tot_batches // 10

  for epoch in range(h['epochs']):
    start = time.time()
    running_loss = 0.0
    running_count = 0
    if bad_loss:
      break
    thrash_rate = 100
    for i, data in enumerate(train_loader):
      global_step += 1
      data = {k: d.to(device) for k,d in data.items()}
      inputs = data['features']
      labels = data['labels']

      optimizer.zero_grad()

      outputs = model(inputs)

      loss = loss_fn(outputs, labels)

      add_loss_weight = h.get('end_of_word_loss_weight', 0)
      if add_loss_weight:
        end_of_word_indices = data['end_of_word_index']
        loss += torch.nn.CrossEntropyLoss()(model.end_word_pred, end_of_word_indices)*add_loss_weight

      loss.backward()
      optimizer.step()
      scheduler.step()

      running_loss += loss.item()
      running_count += 1

      # Log frequency
      if global_step % one_tenth == 0 or global_step < one_tenth and global_step % (one_tenth // 10) == 0 or global_step==1:
        ns_per_example = ((time.time()-start)/(h['batch_size']*(i+1)))*1e6
        if h.get('eval_acc', False):
          accuracy = acc_fn(outputs, labels)
          mt.Track('train', 'accuracy', Torch2Py(accuracy))

        mt.Track('train', 'global_step', global_step)
        mt.Track('train', 'epoch', epoch)

        mt.Track('train', 'loss', Torch2Py(loss.detach()), 3)
        if add_loss_weight:
          mt.Track('train', 'index_loss', Torch2Py(loss.detach()))
        mt.Track('train', 'lr_at_step', scheduler.get_last_lr()[0])
        mt.Track('train', 'us/ex', ns_per_example)
        mt.Track('train', '% cmplt', 100*global_step/tot_batches, -2)
        mt.Log('train')

      if loss.item() > 1000 or torch.isnan(loss):
        bad_loss = True
        logging.info("Loss diverging. quitting. %s" % loss.item())
        break
      running_loss = running_count = 0.0

  return not bad_loss, global_step

class VocabDataset(torch.utils.data.Dataset):
  """Dataset with features: token characters and labels: token index or embedding
     provides a way to add random words after the token of interest and random misspellings.
  """
  def __init__(self, vocab_file, char_to_idx_file, embedding_file, word_length, shuffle=False, 
               misspelling_rate=None, misspelling_transforms=None, misspelling_type=None,
               add_next_word=False, add_random_count=0, space_freq=1.0):
    self.data = self._preprocess(vocab_file, char_to_idx_file, embedding_file, word_length)
    self.keys=['word_char_encoding', 'embeddings']
    self.misspelling_rate =misspelling_rate
    self.misspelling_transforms=misspelling_transforms
    self.misspelling_type=misspelling_type
    self.add_next_word=add_next_word
    self.add_random_count = add_random_count
    self.space_freq = space_freq
    self.embedding_matrix = self.data['embeddings']

    self.data['word_indices'] = torch.arange(len(self.data['embeddings']))
    self.char_to_idx_map = self.data['char_to_idx_map']

    chars_for_insertion = [
         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
         'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    self.insertion_lookup = torch.tensor([self.char_to_idx_map[c] for c in chars_for_insertion])
    self.transform_options = [self.add_letter,
                              self.substitute_letter,
                              self.transpose_letters,
                              self.delete_letter,
                              self.repeat_letter]

  def __len__(self):
    return len(self.data[self.keys[0]])

  def __getitem__(self, idx):
    if self.misspelling_rate == "not set":
      raise "Must set mispelling rate (or explicitly set to None) as a hyperparameter"
    char_encoded_input = self.data['word_char_encoding'][idx]

    item = {
      'labels': self.data['word_indices'][idx].long(),
    }

    if self.misspelling_rate and torch.rand((1,)) <= self.misspelling_rate:
      char_encoded_input = self.apply_misspelling(char_encoded_input)

    end = self.end_of_word(char_encoded_input)
    item['end_of_word_index'] = end
    if torch.rand(size=())<.9: # skip 1/10th of the time
      for i in range(self.add_random_count):
        char_encoded_input =self.add_random_next_word(char_encoded_input, end)
        end = self.end_of_word(char_encoded_input)

    item['features'] = char_encoded_input
    return item

  def GetCharEncoding(self, word, char_to_idx_map, word_length):
    enc = [0]*word_length
    for i, c in enumerate(word):
      if i>= word_length:
        break
      enc[i] = char_to_idx_map.get(c,0)
    return enc

  def _preprocess(self, vocab_file, char_to_idx_file, embedding_file, word_length):
    bert_vocab = []
    with open(vocab_file) as f:
      for l in f.readlines():
        word = l.strip()
        if word.startswith("##"):
          word = word.replace("##", "")
        bert_vocab.append(word)

    bert_emb = torch.load(embedding_file)
    assert len(bert_vocab) == len(bert_emb)
    longest_word_in_bert_vocab = max([len(w) for w in bert_vocab])

    char_to_idx_map = torch.load(char_to_idx_file)

    word_encodings = []
    for word in bert_vocab:
      encoding = self.GetCharEncoding(word.lower(), char_to_idx_map, word_length)
      word_encodings.append(encoding)

    repeat_map = defaultdict(int)
    for w in word_encodings:
      repeat_map[tuple(w)]+=1
    valid_list = [1 if repeat_map[tuple(x)]<=1 else 0 for x in word_encodings]
    word_char_encoding = torch.tensor(word_encodings, dtype=torch.int64)
    valid_list = torch.tensor(valid_list, dtype=torch.int64)
    word_char_encoding = word_char_encoding[torch.where(valid_list==1)]

    embeddings = bert_emb[torch.where(valid_list==1)]
    return {"word_char_encoding":word_char_encoding,
            "embeddings": embeddings,
            "char_to_idx_map":char_to_idx_map}

  def random_location(self, word_size):
    return torch.randint(0, max(word_size, 0), size=())

  def random_letter(self,):
    return self.insertion_lookup[torch.randint(0, self.insertion_lookup.shape[0], size=())]

  def word_len(self, word):
    return (word!=0).sum()

  def add_letter(self, word):
    word = word.clone()
    n = self.random_location(min(word.shape[0], self.word_len(word)+1))
    p2 = word[n:].clone()
    rl = self.random_letter()
    word[n+1:] = p2[:-1] # this will remove the last letter of some words (very few. not concerned for now)
    word[n] = rl
    return word

  def repeat_letter(self, word):
    word = word.clone()
    n = self.random_location(self.word_len(word))
    p2 = word[n:].clone()
    word[n+1:] = p2[:-1] # this will remove the last letter of some words (very few. not concerned for now)
    return word

  def substitute_letter(self, word):
    word = word.clone()
    n = self.random_location(self.word_len(word))
    rl = self.random_letter()
    word[n] = rl
    return word

  def delete_letter(self, word):
    word = word.clone()
    n = self.random_location(self.word_len(word))
    word[n:-1] = word[n+1:].clone()
    word[-1] = 0
    return word

  def transpose_letters(self, word):
    word = word.clone()
    n = self.random_location(self.word_len(word)-1)
    t = word[n].clone()
    word[n] = word[n+1]
    word[n+1] = t
    return word

  def AlterWord(self, word, transforms, misspelling_type=None):
    if self.word_len(word) <= 2:
      return word

    if misspelling_type == "add":
      change_fn = self.add_letter
    if misspelling_type == "repeat":
      change_fn = self.repeat_letter
    if misspelling_type == "substitute":
      change_fn = self.substitute_letter
    if misspelling_type == "delete":
      change_fn = self.delete_letter
    if misspelling_type == "transpose":
      change_fn = self.transpose_letters

    for i in range(transforms):
      if not misspelling_type:
        change_fn = self.transform_options[torch.randint(0, len(self.transform_options), size=())]
      word = change_fn(word)
    return word

  def apply_misspelling(self, char_encoded_input):
    return self.AlterWord(char_encoded_input,
                          transforms=self.misspelling_transforms,
                          misspelling_type=self.misspelling_type
    )

  def end_of_word(self, char_encoded_input):
    chars = char_encoded_input!=0
    return chars.sum(-1)

  def append_single(self, char_encoded_input, add_word, end):
    c_len = len(char_encoded_input)
    if torch.rand(size=())>=self.space_freq:
      end = torch.clamp(end, max=c_len-1)
      char_encoded_input[end:] = self.data['word_char_encoding'][add_word][:c_len-end]
    else:
      char_encoded_input[end:end+1] = self.data['char_to_idx_map'][' '] # space char
      end = torch.clamp(end, max=c_len-1)
      char_encoded_input[end+1:] = self.data['word_char_encoding'][add_word][:c_len-end-1]
    return char_encoded_input
      
  def add_random_next_word(self, char_encoded_input, end):
    # returns the word with a random word added at the end of the word
    char_encoded_input = char_encoded_input.clone()
    if len(char_encoded_input.size())>=2:
      add_word = torch.randint(0, len(self.data['embeddings']), (len(char_encoded_input),))
      new_words=[]
      for i, single_encoded in enumerate(char_encoded_input):
        new_words.append(self.append_single(single_encoded, add_word[i], end[i]))
      char_encoded_input = torch.stack(new_words)
    else:
      add_word = torch.randint(0, len(self.data['embeddings']), ())
      char_encoded_input = self.append_single(char_encoded_input, add_word, end)

    return char_encoded_input

class LambdaLayer(nn.Module):
  def __init__(self, lambd):
    super(LambdaLayer, self).__init__()
    self.lambd = lambd
  def forward(self, x):
    return self.lambd(x)

def gelu(x):
  return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

class ConvSegment(nn.Module):
  def __init__(self, input_channels, activation, kernel_filters_sizes):
    super(ConvSegment, self).__init__()
    if activation == 'relu':
      self.conv_activation = nn.ReLU()
    if activation == 'gelu':
      self.conv_activation = gelu
    elif activation == 'sigmoid':
      self.conv_activation = nn.Sigmoid()

    self.convs = []

    last_out_chan = input_channels
    for k, f in kernel_filters_sizes:
      self.convs.append(nn.Conv1d(last_out_chan, f, k))
      self.convs.append(self.conv_activation)
      last_out_chan = f
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
    return unfolded.flatten(-2).permute(0,2,1)

class TokensToEmb(nn.Module):
  def __init__(self, h):
    super(TokensToEmb, self).__init__()
    self.sequence_len = h['word_length']
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
    self.vocab_project = nn.Conv1d(seg2_out_size, h['token_vocab_size'], 1)


    # this probs isnt necessary - but if i decide to change the shapes it might be useful later.
    self.end_of_word_weight = h.get('end_of_word_loss_weight', 0)
    if self.end_of_word_weight:
      self.end_word_dense = torch.nn.Linear(h['word_length'], h['word_length'])
      self.end_word_softmax = torch.nn.Softmax(-1)

  def forward_first_half(self, x):
    embeddings  = self.emb(x).permute(0,2,1)

    # end of word attention heuristic
    if self.end_of_word_weight:
      pass 
      # Hardcode attention
      # bs = len(x)
      # attn = torch.zeros(bs, self.sequence_len, device='cuda', requires_grad=False)
      # attn[torch.arange(bs), eow] = 1.0
      # attn = attn.unsqueeze(-1)

      # Learn attention
      # seg_attn_out = self.segment_attn(embeddings)
      # self.end_word_pred = self.end_word_softmax(self.end_word_dense(seg_attn_out.flatten(-2)))

      # embeddings_attn = torch.cat([embeddings, seg_attn_out.permute(0,2,1)], dim=-2)
      # embeddings_attn = torch.cat([embeddings, attn.permute(0,2,1)], dim=-2)

    seg1_out = self.segment1(embeddings)
    return seg1_out

  def forward_second_half(self, x):
    seg2_out = self.segment2(x)
    logits = self.vocab_project(seg2_out).permute(0,2,1)
    return logits

  def forward(self, x):
    # Only generate 1 embedding no matter what the first few layers of conv produce
    h1 = self.forward_first_half(x)
    embedding_out = self.forward_second_half(h1)
    return embedding_out.squeeze(1)

class Net(nn.Module):
  def __init__(self, h):
    super(Net, self).__init__()
    self.tokens_to_emb = TokensToEmb(h)
  def forward(self, x):
    return self.tokens_to_emb(x)

def InitDataset(exp_info, dev='cuda'):
  # load and process on cpu and load each batch to GPU in the training loop
  ds = VocabDataset(exp_info['vocab_file'],
                    exp_info['char_to_idx_file'],
                    exp_info['embedding_file'],
                    exp_info['word_length'],
                    misspelling_rate="not set")

  # hacky to pass data dependent model hyperparams
  exp_info['data_info'] = {}
  exp_info['data_info']['char_vocab_size'] = len(ds.char_to_idx_map)
  exp_info['data_info']['token_vocab_size'] = len(ds.embedding_matrix)
  
  logging.debug("n_train_examples: %s" % len(ds))

  return ds

def CreateModel(h):
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

def SetDatasetHyperParams(data,h):
  data.misspelling_rate = h['misspelling_rate']
  data.misspelling_transforms = h['misspelling_transforms']
  data.add_random_count = h.get('add_random_count', 0)
  data.space_freq = h.get('space_freq', 1)

def RunOne(h, model, data, mt, dev='cuda'):
  model.to(dev)

  SetDatasetHyperParams(data,h)

  if 'seed' in h:
    torch.manual_seed(h['seed'])

  train_loader = DataLoader(
    data, batch_size=h['batch_size'], shuffle=True, num_workers=os.cpu_count())
  validation_loader = DataLoader(
    data, batch_size=1000, shuffle=False, num_workers=os.cpu_count())
  logging.debug("n_train_batches: %s" % (len(train_loader.dataset)//h['batch_size']))

  logging.info("Experiment Model:\n" + str(model))
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

  if h['optimizer'] == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=h['learning_rate'], momentum=h['momentum'])
  if h['optimizer'] == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=h['learning_rate'])
  if h['loss_fn'] == 'xent':
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
