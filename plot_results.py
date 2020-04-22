#! /bin/python

import os
from yaml import safe_load
import sys
import numpy as np
import hiplot as hip
import copy

def GetFlatInfoResults(exp, hashes):
  flat = {}
  flat.update(GetFinalVals(exp))
  flat.update(GetHyperParams(exp, hashes))
  return flat

def GetHyperParams(results, seen_hashes):
  h = copy.deepcopy(results['exp_info']['hyperparameters'])
  lists = []
  expanded = {}
  expanded['hash'] = results['exp_info']['experiment_hash']
  expanded['uid'] = results['exp_info']['experiment_hash']
  
  cp = h.get('model_checkpoint', None)
  if cp and cp.split('/')[-1] in seen_hashes:
    expanded['from_uid'] = cp.split('/')[-1]
  for k, v in h.items():
    if type(v)== bool:
      h[k] = int(v)
    if type(v) == list:
      lists.append(k)
      names = [k]
      pref = False
      if '.' in k:
        pre, k = k.split(".")
        pref = True
      if "|" in k:
        names = k.split("|")
        if pref:
          names = [pre+"."+n for n in names]
      count = 1
      for val in v:
        for i, name in enumerate(names):
          expanded[name+str(count)] = val[i] if len(names)>1 else val
        count+=1
  for l in lists:
    del h[l]
  h.update(expanded)
  return h

def GetFinalVals(results):
  val_results = results['val']
  train_results = results['train']
  finals = {}
  if 'us/ex' in train_results:
    finals['train_time'] = np.mean(train_results['us/ex'])
  for k,v in val_results.items():
    finals[k] = v[-1]

  for k,v in results['exp_info'].items():
    if k=='hyperparameters':
      continue
    finals[k] = v
  return finals

def normalize(flats):
  keys = set()
  for f in flats:
    keys.update(f.keys())
  for f in flats:
    for k in keys:
      if not k in f and k!='from_uid':
        f[k] = 0
      if not k in f and k=='from_uid':
        f[k] = None

if __name__ == "__main__":
  experiment_dir = sys.argv[1]

  results = []
  result_files = []

  result_dir = experiment_dir + '/results'
  hashes = os.listdir(result_dir)
  result_files = [os.path.join(result_dir, fname) for fname in hashes]

  for fname in result_files:
    with open(fname, 'r') as f:
      yaml = safe_load(f.read())
      if not yaml:
        print("empty:", f)
      results.append(yaml)

  results = [r for r in results if not 'exit_info' in r]
  flats = [GetFlatInfoResults(x, hashes) for x in results if 'xent_loss' in x['val']]

  normalize(flats)
  exp_vis = hip.Experiment.from_iterable(flats)
  html = exp_vis.to_html('/tmp/hiplot.html')
