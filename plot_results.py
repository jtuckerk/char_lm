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

EXPAND_LISTS=False
def IncludeLists(list_names, expanded_dict, name, hp_list):
  list_names.append(name)
  names = [name]
  pref = False
  if '.' in name:
    pre, name = name.split(".")
    pref = True
  if "|" in name:
    names = name.split("|")
    if pref:
      names = [pre+"."+n for n in names]
  count = 1
  if not EXPAND_LISTS:
    # just include the number of list elements to keep the graphs from getting cluttered
    expanded_dict[name] = len(hp_list)
  else:
    for val in hp_list:
      for i, name in enumerate(names):
        expanded_dict[name+str(count)] = val[i] if len(names)>1 else val
      count+=1

def GetHyperParams(results, seen_hashes):
  h = copy.deepcopy(results['exp_info']['hyperparameters'])
  lists = []
  expanded = {}
  expanded['hash'] = results['exp_info']['experiment_hash']
  expanded['uid'] = results['exp_info']['experiment_hash']
  
  cp = h.get('model_checkpoint', None)
  if cp and cp.split('/')[-1] in seen_hashes:
    expanded['from_uid'] = cp.split('/')[-1]

  # Deal with hyperparams other than ints/floats
  for k, v in h.items():
    if type(v)== bool:
      h[k] = int(v)
    elif type(v) == list:
      IncludeLists(lists, expanded, k, v)
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
  print("%s result files found" % len(result_files))
  for fname in result_files:
    with open(fname, 'r') as f:
      yaml = safe_load(f.read())
      if not yaml:
        print("empty:", f)
      results.append(yaml)

  results = [r for r in results if not 'exit_info' in r]
  flats = [GetFlatInfoResults(x, hashes) for x in results]

  normalize(flats)
  exp_vis = hip.Experiment.from_iterable(flats)
  html = exp_vis.to_html('/tmp/hiplot.html')
