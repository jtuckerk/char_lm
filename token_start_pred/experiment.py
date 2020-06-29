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
import torch.nn.functional as F
import os
import math

level = logging.getLevelName("INFO")
logging.basicConfig(
  level=level,
  format="[%(asctime)s] %(message)s",
  datefmt="%H:%M:%S",
)

Torch2Py = lambda x: x.cpu().numpy().tolist()

def bin_acc_fn(outputs, labels):
  binary = outputs>.5
  right = binary==labels
  return right.float().mean()

BCE_loss = nn.BCELoss()
def bce_loss_fn(outputs, labels):
  return BCE_loss(outputs, labels)

def Validate(val_loader, model, global_step, h, metric_tracker, val_type):
  cum_acc = 0
  cum_bce_loss = 0

  for i, data in enumerate(val_loader):
    inputs = data['char_encoded_seqs']
    labels = data['token_start_positions']

    with torch.no_grad():
      outputs = model(inputs)
      bce_loss = bce_loss_fn(outputs, labels)
      cum_bce_loss += bce_loss

      acc = bin_acc_fn(outputs, labels)
      cum_acc += acc

  steps = i+1
  metric_tracker.Track(val_type, 'global_step', global_step)
  metric_tracker.Track(val_type, 'accuracy', Torch2Py(cum_acc/steps))
  metric_tracker.Track(val_type, 'bce_loss', Torch2Py(cum_bce_loss.detach()/steps))

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
      inputs = data['char_encoded_seqs']
      labels = data['token_start_positions']

      optimizer.zero_grad()

      outputs = model(inputs)

      loss = loss_fn(outputs, labels)

      loss.backward()
      optimizer.step()
      scheduler.step()

      running_loss += loss.item()
      running_count += 1

      # Log frequency
      if global_step % one_tenth == 0 or global_step < one_tenth and global_step % (one_tenth // 10) == 0 or global_step==1:
        ns_per_example = ((time.time()-start)/(h['batch_size']*(i+1)))*1e6
        accuracy = bin_acc_fn(outputs, labels)

        mt.Track('train', 'accuracy', Torch2Py(accuracy))
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

class TokenStartDataset(torch.utils.data.Dataset):
  """Dataset with features: int encoded characters and labels: token start index position
  """
  def __init__(self, data_file, device='cuda'):
    self.data = torch.load(data_file)
    self.data['char_encoded_seqs'] = self.data['chars_encoded'].to(device)
    # if the dataset has start and end indexes - convert to idx 1-marked.
    if len(self.data['token_start_offsets'].shape) == 3:
      starts = self.data['token_start_offsets'][:, :, 0]
      seq_enc_starts = torch.zeros(self.data['char_encoded_seqs'].shape).to(starts.device)
      seq_enc_starts[torch.arange(self.data['char_encoded_seqs'].shape[0]).view(-1,1), starts.long()] = 1
      self.data['token_start_positions'] = seq_enc_starts.to(device)
      print("2part offset")
    else:
      self.data['token_start_positions'] = self.data['token_start_offsets']
      print("1hot encoding ish")
      
    assert len(self.data['char_encoded_seqs']) == len(self.data['token_start_positions'])
    
  def __len__(self):
    return len(self.data['char_encoded_seqs'])

  def __getitem__(self, idx):
    return {
      'char_encoded_seqs': self.data['char_encoded_seqs'][idx].long(),
      'token_start_positions': self.data['token_start_positions'][idx].float()
    }

class LambdaLayer(nn.Module):
  def __init__(self, lambd):
    super(LambdaLayer, self).__init__()
    self.lambd = lambd
  def forward(self, x):
    return self.lambd(x)

def gelu(x):
  return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

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

class TokStartAttn(nn.Module):
  def __init__(self, h):
    super(TokStartAttn, self).__init__()
    self.emb = nn.Embedding(h['char_vocab_size'], h['char_embedding_size'])

    self.char_input_window_size = h['conv.kernel|filter_sizes'][0][0]
    self.conv = ConvSegment(h['char_embedding_size'], h['conv_activation'], h['conv.kernel|filter_sizes'], True)
    conv_out_size = h['conv.kernel|filter_sizes'][-1][1]

    # no nonlinearity
    self.final_conv = nn.Conv1d(conv_out_size, 1, 1)

  def forward(self, x):

    embout = self.emb(x).permute(0,2,1)
    char_block = self.conv(embout)
    char_block_logits = self.final_conv(char_block).permute(0,2,1).squeeze(-1)

    return char_block_logits
    
class Net(nn.Module):
  def __init__(self, h):
    super(Net, self).__init__()
    self.tokens_start_attn = TokStartAttn(h)
    
  def forward(self, x):
    return torch.nn.Sigmoid()(self.tokens_start_attn(x))

def InitDataset(exp_info, dev='cuda'):
  # load and process on cpu and load each batch to GPU in the training loop
  ds = TokenStartDataset(exp_info['dataset_file'], dev)
  tr, va = exp_info['dataset_split'] # the rest is test
  n_train_examples, n_val_examples = int(len(ds)*tr), int(len(ds)*va)
  n_test_examples = len(ds)- n_train_examples - n_val_examples
  train_d, val_d, test_d = torch.utils.data.random_split(
    ds, [n_train_examples,n_val_examples,n_test_examples])
  logging.debug("n_train_examples: %s" % n_train_examples)
  logging.debug("n_val_examples: %s" % n_val_examples)

  return train_d, val_d, test_d

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
  train_d, val_d, test_d = data
  model.to(dev)

  if 'seed' in h:
    torch.manual_seed(h['seed'])

  train_loader = DataLoader(train_d, batch_size=h['batch_size'], shuffle=True)
  validation_loader = DataLoader(val_d, batch_size=h['batch_size'], shuffle=False)
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
  if h['loss_fn'] == 'bce':
    loss_fn = bce_loss_fn

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
