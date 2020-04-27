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
import math

level = logging.getLevelName("INFO")
logging.basicConfig(
  level=level,
  format="[%(asctime)s] %(message)s",
  datefmt="%H:%M:%S",
)

# TODO
# Incorporate mask into loss so that embeddings with norm 0 can be predicted for non-word chars

Torch2Py = lambda x: x.cpu().numpy().tolist()

xent = nn.CrossEntropyLoss()
def xent_loss_fn(output, labels):
  return xent(output.view(-1, output.size(-1)), labels.view(-1))

# returns char acc and word acc
def acc_fn(output, labels):
  top = output.argmax(-1)
  right = top==labels
  char_acc = right.float().mean()
  word_acc = (right.sum(-1)==right.shape[-1]).float().mean()
  return char_acc, word_acc

def Validate(val_loader, model, global_step, h, metric_tracker, val_type):
  cum_char_acc = 0
  cum_word_acc = 0
  cum_xent_loss = 0

  correct_words = []
  for i, data in enumerate(val_loader):
    inputs = data['features']
    labels = data['char_labels']

    with torch.no_grad():
      outputs = model(inputs)
      xent_loss = xent_loss_fn(outputs, labels)
      cum_xent_loss += xent_loss

      char_acc, word_acc = acc_fn(outputs, labels)
      cum_char_acc += char_acc
      cum_word_acc += word_acc

  steps = i+1
  metric_tracker.Track(val_type, 'global_step', global_step)
  metric_tracker.Track(val_type, 'char_acc', Torch2Py(cum_char_acc/steps))
  metric_tracker.Track(val_type, 'word_acc', Torch2Py(cum_word_acc/steps))
  metric_tracker.Track(val_type, 'xent_loss', Torch2Py(cum_xent_loss.detach()/steps))

def Train(train_loader, model, loss_fn, optimizer, h, mt):
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
      inputs = data['features']
      labels = data['char_labels']

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
        char_acc, word_acc = acc_fn(outputs, labels)
        mt.Track('train', 'char_acc', Torch2Py(char_acc))
        mt.Track('train', 'word_acc', Torch2Py(word_acc))
        mt.Track('train', 'global_step', global_step)
        mt.Track('train', 'epoch', epoch)
        mt.Track('train', 'loss', Torch2Py(loss.detach()), 3)
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

class ReverseVocabDataset(torch.utils.data.Dataset):
  """Dataset with features: token embeddings and labels: characters
  """
  def __init__(self, vocab_file, char_to_idx_file, embedding_file, word_length):
    self.data = self._preprocess(vocab_file, char_to_idx_file, embedding_file, word_length)
    self.data['word_indices'] = torch.arange(len(self.data['embeddings'])).cuda()
    self.char_to_idx_map = self.data['char_to_idx_map']


  def __len__(self):
    return len(self.data['word_char_encoding'])

  def __getitem__(self, idx):
    char_encoded_input = self.data['word_char_encoding'][idx]
    mask = char_encoded_input!=0
    item = {
      'char_labels': char_encoded_input.long(),
      'tok_labels': self.data['word_indices'][idx],
      'mask': mask,
      'features': self.data['embeddings'][idx],
    }

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
    return {"word_char_encoding":word_char_encoding.cuda(),
            "embeddings": embeddings.cuda(),
            "char_to_idx_map":char_to_idx_map}


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
      Gelu = LambdaLayer(gelu)
      self.conv_activation = Gelu
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


class EmbToTokens(nn.Module):
  def __init__(self, h):
    super(EmbToTokens, self).__init__()

    self.conv = ConvSegment(h['token_embedding_size'], h['conv_activation'], h['conv.kernel|filter_sizes'])
    conv_out_size = h['conv.kernel|filter_sizes'][-1][1]
    self.word_len, self.char_vocab_size = h['word_length'], h['char_vocab_size']
    self.final_conv = nn.Conv1d(conv_out_size, h['word_length']*h['char_vocab_size'], 1)

  def forward(self, x):
    convout = self.conv(x)
    return self.final_conv(convout).reshape(-1, self.word_len, self.char_vocab_size)

class Net(nn.Module):
  def __init__(self, h):
    super(Net, self).__init__()
    self.emb_to_toks = EmbToTokens(h)
    
  def forward(self, x):
    # batchXChannelsXseq
    x=x.unsqueeze(-1)
    return self.emb_to_toks(x)

def InitDataset(exp_info, dev='cuda'):
  # load and process on cpu and load each batch to GPU in the training loop
  ds = ReverseVocabDataset(exp_info['vocab_file'],
                           exp_info['char_to_idx_file'],
                           exp_info['embedding_file'],
                           exp_info['word_length'])
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


def RunOne(h, model, data, mt, dev='cuda'):
  model.to(dev)

  if 'seed' in h:
    torch.manual_seed(h['seed'])

  train_loader = DataLoader(data, batch_size=h['batch_size'], shuffle=True)
  validation_loader = DataLoader(data, batch_size=1000, shuffle=False)
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
      mt.AddExpInfo('size_params', size_params)
      mt.AddExpInfo('size_bytes', size_bytes)
      mt.AddExpInfo('exit_info', 'size outside of range. skipping')
      return model
              

  if h['optimizer'] == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=h['learning_rate'], momentum=h['momentum'])
  if h['optimizer'] == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=h['learning_rate'])
  if h['loss_fn'] == 'xent':
    loss_fn = xent_loss_fn

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
