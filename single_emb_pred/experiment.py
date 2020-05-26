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
import torch.nn.functional as F

# TODO
# nearest neighbor acc optimization

MSE = nn.MSELoss()

def mse_loss_fn(outputs, labels, embedding_matrix):
  return MSE(outputs, embedding_matrix[labels])

def squared_mse_loss_fn(outputs, labels, embedding_matrix):
  return torch.sqrt(MSE(outputs, embedding_matrix[labels]))

# simplified version of https://arxiv.org/pdf/1812.04616.pdf -- no bessel func, no regularization terms
def vmf_simp_loss(out, labels, embedding_matrix):
  targets = embedding_matrix[labels]
  normalization_const = out.norm(dim=-1)
  # batchwise matrix multiply
  likelihood = out.unsqueeze(1)@targets.view(-1,targets.shape[-1], 1)
  return (-torch.log(normalization_const) - likelihood.squeeze()).mean()

def cos_loss(out, labels, embedding_matrix):
  targets = embedding_matrix[labels]
  pred_normalize = out.norm(dim=-1)
  target_normalize = targets.norm(dim=-1)
  # batchwise matrix multiply
  inner_prod = out.unsqueeze(1)@targets.view(-1,targets.shape[-1], 1)
  normed = inner_prod.squeeze()/(pred_normalize*target_normalize)
  return (-1*normed).mean()

def exp_cos_loss(out, labels, embedding_matrix):
  targets = embedding_matrix[labels]
  pred_normalize = out.norm(dim=-1)
  target_normalize = targets.norm(dim=-1)
  # batchwise matrix multiply
  inner_prod = out.unsqueeze(1)@targets.view(-1,targets.shape[-1], 1)
  normed = inner_prod.squeeze()/(pred_normalize*target_normalize)
  return (-1*torch.exp(normed)).mean()

def vmf_simp_loss_neg_sample(out, labels, embedding_matrix):
  targets = embedding_matrix[labels]
  negative_examples = embedding_matrix[torch.randperm(len(embedding_matrix))[:len(labels)]]
  normalization_const = out.norm(dim=-1)
  # batchwise matrix multiply
  likelihood = out.unsqueeze(1)@targets.view(-1,targets.shape[-1], 1)
  neg_likelihood = out.unsqueeze(1)@negative_examples.view(-1,targets.shape[-1], 1)
  return (-torch.log(normalization_const)+neg_likelihood.squeeze() - likelihood.squeeze()).mean()

def vmf_simp_loss_neg_sample_reg1(out, labels, embedding_matrix):
  targets = embedding_matrix[labels]
  negative_examples = embedding_matrix[torch.randperm(len(embedding_matrix))[:len(labels)]]
  normalization_const = out.norm(dim=-1)
  # batchwise matrix multiply
  likelihood = out.unsqueeze(1)@targets.view(-1,targets.shape[-1], 1)
  neg_likelihood = out.unsqueeze(1)@negative_examples.view(-1,targets.shape[-1], 1)
  return (-torch.log(normalization_const)+neg_likelihood.squeeze() - likelihood.squeeze() + normalization_const).mean()

# not actually fast... can run on colab GPU with lots of mem but doesn't speed up much
def mse_entropy_fast(outputs, labels, embedding_matrix):
  mse = torch.nn.MSELoss(reduction='none')
  emb_mat =embedding_matrix.unsqueeze(1).expand(-1,len(outputs),-1)
  diffs = mse(emb_mat, outputs)

  logits = diffs.mean(-1).permute(1,0)

  loss = xent_loss(logits, labels)
  return loss

Torch2Py = lambda x: x.cpu().numpy().tolist()

xent_loss = nn.CrossEntropyLoss()
def dot_entropy_loss_fn(logits, labels):
  loss = xent_loss(logits, labels)
  return loss

def dot_acc_fn(logits, labels, emb):
  top = logits.argmax(-1)
  right = top==labels
  return right.float().mean()

def cosine_similarity_acc_fn(outputs, labels, embedding_matrix):
  out_norm = outputs.norm(dim=-1)
  emb_norm = embedding_matrix.norm(dim=-1)
  inner_prods = outputs@embedding_matrix.T
  n1 = inner_prods/out_norm.unsqueeze(1)
  n2 = n1/emb_norm

  preds = n2.argmax(-1)
  right = labels==preds
  return right.float().mean()

def nearest_neighbor_acc_fn(outputs, labels, embedding_matrix):
  # overflows memory
  # emb_mat =embedding_matrix.unsqueeze(1).expand(-1,len(outputs),-1)
  # diffs = ((emb_mat - outputs)**2)
  # mse = diffs.mean(-1).permute(1,0)

  # too slow
  # MSE = torch.nn.MSELoss()
  # mins = []
  # for pred in outputs:
  #   vals = []
  #   for emb in embedding_matrix:
  #     vals.append(MSE(emb, pred))
  #   mse = torch.stack(vals)

  # just right
  mins = []
  mse = torch.nn.MSELoss(reduction='none')
  for pred in outputs:
    # repeat the pred and compare it to each entry in the embedding matrix
    tiled_pred = pred.unsqueeze(0).expand(len(embedding_matrix), -1)
    mins.append(mse(tiled_pred, embedding_matrix).mean(-1).argmin())
  min_vec = torch.stack(mins)
  right = (min_vec == labels)
  acc = right.float().mean()

  return acc

def Validate(val_loader, model, global_step, h, metric_tracker, val_type):
  device = next(model.parameters()).device

  cum_nearest_neighbor_acc = 0
  cum_mse_loss = 0
  cum_cos_loss = 0
  cum_vmf_loss = 0  
  cum_dot_acc = 0
  cum_cos_acc = 0  
  cum_dot_xent_loss = 0

  embedding_matrix = val_loader.embedding_matrix

  correct_words = []
  for i, data in enumerate(val_loader):
    data = {k: d.to(device) for k,d in data.items()}
    inputs = data['features'].to(device)
    labels = data['labels'].long()
    target_embeddings = data['target_embeddings']

    with torch.no_grad():
      outputs = model(inputs)
      cum_vmf_loss += vmf_simp_loss(outputs, labels, embedding_matrix)
      mse_loss = mse_loss_fn(outputs, labels, embedding_matrix)
      cum_cos_loss += cos_loss(outputs, labels, embedding_matrix)      
      logits = torch.matmul(outputs, embedding_matrix.T)
      cum_dot_xent_loss += dot_entropy_loss_fn(logits, labels)
      cum_mse_loss += mse_loss

      if h.get('eval_acc', False):
        nearest_neighbor_acc = nearest_neighbor_acc_fn(outputs, labels, embedding_matrix)
        cum_nearest_neighbor_acc += nearest_neighbor_acc
        cum_dot_acc += dot_acc_fn(logits, labels, embedding_matrix)
        cum_cos_acc += cosine_similarity_acc_fn(outputs, labels, embedding_matrix)

  steps = i+1
  metric_tracker.Track(val_type, 'global_step', global_step)
  if h.get('eval_acc', False):
    metric_tracker.Track(val_type, 'accuracy', Torch2Py(cum_nearest_neighbor_acc/steps))
    metric_tracker.Track(val_type, 'dot_acc', Torch2Py(cum_dot_acc/steps))
    metric_tracker.Track(val_type, 'cos_acc', Torch2Py(cum_cos_acc/steps))
  metric_tracker.Track(val_type, 'vmf_simp_loss', Torch2Py(cum_vmf_loss.detach()/steps))
  metric_tracker.Track(val_type, 'mse_loss', Torch2Py(cum_mse_loss.detach()/steps))
  metric_tracker.Track(val_type, 'cos_loss', Torch2Py(cum_cos_loss.detach()/steps))
  metric_tracker.Track(val_type, 'dot_xent_loss', Torch2Py(cum_dot_xent_loss.detach()/steps))

def Train(train_loader, model, loss_fn, optimizer, h, mt):

  device = next(model.parameters()).device

  n_batches = int(len(train_loader.dataset)/h['batch_size'])
  embedding_matrix = train_loader.embedding_matrix

  logging.info("TRAINING")
  tot_batches = n_batches*h['epochs']
  logging.info("total batches: %s" % tot_batches)
  global_step = 0

  acc_fn = nearest_neighbor_acc_fn

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
      target_embeddings = data['target_embeddings']

      optimizer.zero_grad()

      outputs = model(inputs)

      loss = loss_fn(outputs, labels, embedding_matrix)

      mse_loss_weight = h.get('mse_loss_weight', 0)
      if mse_loss_weight:
        main_loss = loss.clone()
        mse_loss = mse_loss_fn(outputs, labels, embedding_matrix)*mse_loss_weight
        loss+= mse_loss
      dot_loss_weight = h.get('dot_loss_weight', 0)
      if dot_loss_weight:
        logits = torch.matmul(outputs, embedding_matrix.T)
        dot_loss = dot_entropy_loss_fn(logits, labels)*dot_loss_weight
        loss+= dot_loss
        
      loss.backward()
      optimizer.step()
      scheduler.step()

      running_loss += loss.item()
      running_count += 1

      # Log frequency
      if global_step % one_tenth == 0 or global_step < one_tenth and global_step % (one_tenth // 10) == 0 or global_step==1:
        ns_per_example = ((time.time()-start)/(h['batch_size']*(i+1)))*1e6
        if h.get('eval_acc', False):
          accuracy = acc_fn(outputs, labels, embedding_matrix)
          mt.Track('train', 'accuracy', Torch2Py(accuracy))

        mt.Track('train', 'gl_step', global_step)
        mt.Track('train', 'epoch', epoch)

        mt.Track('train', 'loss', Torch2Py(loss.detach()), 3)
        if mse_loss_weight:
          mt.Track('train', 'mse_loss', Torch2Py(mse_loss.detach()))
          mt.Track('train', 'main_loss', Torch2Py(main_loss.detach()))
        if dot_loss_weight:
          dot_acc = dot_acc_fn(logits, labels, embedding_matrix)
          mt.Track('train', 'dot_loss', Torch2Py(dot_loss.detach()))     
          mt.Track('train', 'dot_acc', Torch2Py(dot_acc.detach()))     
          
        mt.Track('train', 'lr', scheduler.get_last_lr()[0], 4)
        mt.Track('train', 'us/ex', ns_per_example)
        mt.Track('train', '% cmplt', 100*global_step/tot_batches, -1)
        # track the norm of the gradients over time
        # for name, w in model.named_parameters():
        #   if 'weight' in name:
        #     name = name.split('.')[1:][-3:-1]
        #     name = ".".join(name)
        #     name = name + " "*(6-len(name))
        #     mt.Track('train', name, w.grad.norm().item())
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
      'target_embeddings': self.data['embeddings'][idx],
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
    valid_list = [1]*len(bert_vocab)
    for i, w in enumerate(word_encodings):
      repeat_map[tuple(w)]+=1
      if repeat_map[tuple(w)]>1:
        valid_list[i] = 0

    #ignore any words with duplicates (not using) keeping first instead (some very common words would be discarded otherwise)
    #valid_list = [1 if repeat_map[tuple(x)]<=1 else 0 for x in word_encodings]
    word_char_encoding = torch.tensor(word_encodings, dtype=torch.int64)
    valid_list = torch.tensor(valid_list, dtype=torch.int64)
    word_char_encoding = word_char_encoding[torch.where(valid_list==1)]
    with open('tmp_vocab.txt', 'w') as f:
      f.write("\n".join([bert_vocab[i] for i in Torch2Py(torch.where(valid_list==1)[0])]))

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

def PrintSize(self, input, output):
  print(input[0].shape, "-->", output.shape)
  print((input[0]!=output).sum())

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
  train_loader.embedding_matrix = data.embedding_matrix.to(dev)
  validation_loader.embedding_matrix = train_loader.embedding_matrix
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
  if h['loss_fn'] == 'mse':
    loss_fn = mse_loss_fn
  if h['loss_fn'] == 'squared_mse':
    loss_fn = squared_mse_loss_fn
  if h['loss_fn'] == 'vmf_simplified':
    loss_fn = vmf_simp_loss
  if h['loss_fn'] == 'vmf_simplified_neg_sample':
    loss_fn = vmf_simp_loss_neg_sample
  if h['loss_fn'] == 'vmf_simplified_neg_sample_reg1':
    loss_fn = vmf_simp_loss_neg_sample_reg1
  if h['loss_fn'] == 'cos':
    loss_fn = cos_loss
  if h['loss_fn'] == 'exp_cos':
    loss_fn = exp_cos_loss

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
