"""
Preprocess a raw json dataset into hdf5 and json files for use in lib/loaders

Input: refer loader
Output: json file has
- refs:       [{ref_id, ann_id, box, image_id, split, category_id, sent_ids, att_wds}]
- images:     [{image_id, ref_ids, file_name, width, height, h5_id}]
- anns:       [{ann_id, category_id, image_id, box, h5_id}]
- sentences:  [{sent_id, tokens, h5_id}]
- word_to_ix: {word: ix}
- att_to_ix : {att_wd: ix}
- att_to_cnt: {att_wd: cnt}
- label_length: L

Output: hdf5 file has
/labels is (M, seq_length) int32 array of encoded labels, zeros padded in the end
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import argparse
import string
import os.path as osp
import operator
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize

forbidden_att = ['none', 'other', 'sorry', 'pic', 'extreme', 'rightest', 'tie', 'leftest', 'hard', 'only', 
'darkest', 'foremost', 'topmost', 'leftish','utmost', 'lemon', 'good', 'hot', 'more', 'least', 'less', 
'cant', 'only', 'opposite', 'upright', 'lightest', 'single', 'touching', 'bad', 'main', 'remote', '3pm', 
'same', 'bottom', 'middle']
forbidden_verb = ['none', 'look', 'be', 'see', 'have', 'head', 'show', 'strip', 'get', 'turn', 'wear', 
'reach', 'get', 'cross', 'turn', 'point', 'take', 'color', 'handle', 'cover', 'blur', 'close', 'say', 'go', 
'dude', 'do', 'let', 'think', 'top', 'head', 'take', 'that', 'say', 'carry', 'man', 'come', 'check', 'stuff', 
'pattern', 'use', 'light', 'follow', 'rest', 'watch', 'make', 'stop', 'arm', 'try', 'want', 'count', 'lead', 
'know', 'mean', 'lap', 'moniter', 'dot', 'set', 'cant', 'serve', 'surround', 'isnt', 'give', 'click']
forbidden_noun = ['none', 'picture', 'pic', 'screen', 'background', 'camera', 'edge', 'standing', 'thing', 
'holding', 'end', 'view', 'bottom', 'center', 'row', 'piece']

def build_vocab(refer, params):
  """
  Our vocabulary will add __background__, COCO categories, <UNK>, PAD, BOS, EOS
  """
  # remove bad words, and return final sentences (sent_id -> final)
  count_thr = params['word_count_threshold']
  sentToTokens = refer.sentToTokens

  # count up the number of words
  word2count = {}
  for sent_id, tokens in sentToTokens.items():
    for wd in tokens:
      word2count[wd] = word2count.get(wd, 0) + 1

  # print some stats
  total_words = sum(word2count.values())
  bad_words = [wd for wd, n in word2count.items() if n <= count_thr]
  good_words= [wd for wd, n in word2count.items() if n > count_thr]
  bad_count = sum([word2count[wd] for wd in bad_words])
  print('number of good words: %d' % len(good_words))
  print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(word2count), len(bad_words)*100.0/len(word2count)))
  print('number of UNKs in sentences: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))
  vocab = good_words

  # add category words
  category_names = list(refer.Cats.values()) + ['__background__']
  for cat_name in category_names:
    for wd in cat_name.split():
      if wd not in word2count or word2count[wd] <= count_thr:
        word2count[wd] = 1e5
        vocab.append(wd)
        print('category word [%s] added to vocab.' % wd)

  # add UNK, BOS, EOS, PAD
  if bad_count > 0:
    vocab.append('<UNK>')
  vocab.append('<BOS>')
  vocab.append('<EOS>')
  vocab.insert(0, '<PAD>')  # add PAD to the very front

  # lets now produce final tokens
  sentToFinal = {}
  for sent_id, tokens in sentToTokens.items():
    final = [wd if word2count[wd] > count_thr else '<UNK>' for wd in tokens]
    sentToFinal[sent_id] = final

  return vocab, sentToFinal

def check_sentLength(sentToFinal):
  sent_lengths = {}
  for sent_id, tokens in sentToFinal.items():
    nw = len(tokens)
    sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
  max_len = max(sent_lengths.keys())
  print('max length of sentence in raw data is %d' % max_len)
  print('sentence length distribution (count, number of words):')
  sum_len = sum(sent_lengths.values())
  acc = 0  # accumulative distribution
  for i in range(max_len+1):
    acc += sent_lengths.get(i, 0)
    print('%2d: %10d %.3f%% %.3f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0)*100.0/sum_len, acc*100.0/sum_len))

def encode_captions(sentences, wtoi, params):
  """
  sentences = [{sent_id, tokens, h5_id}]
  """
  max_length = params['max_length']
  M = len(sentences)
  L = np.zeros((M, max_length), dtype=np.int32)
  for i, sent in enumerate(sentences):
    h5_id = sent['h5_id']
    assert h5_id == i
    tokens = sent['tokens']
    for j, w in enumerate(tokens):
      if j < max_length:
        L[h5_id, j] = wtoi[w]
  return L

def check_encoded_labels(sentences, labels, itow):
  for sent in sentences:
    # gd truth 
    print('gd-truth: %s' % (' '.join(sent['tokens'])))
    # deocde labels
    h5_id = sent['h5_id']
    label = labels[h5_id].tolist()
    decoded = ' '.join([itow[i] for i in label if i != 0])
    print('decoded : %s' % decoded)
    print('\n')

def prepare_json(refer, sentToFinal, ref_to_att_wds, params):
  # prepare refs = [{ref_id, ann_id, image_id, box, split, category_id, sent_ids}]
  refs = []
  for ref_id, ref in refer.Refs.items():
    box = refer.refToAnn[ref_id]['bbox']
    att_wds = ref_to_att_wds[ref_id] if ref_id in ref_to_att_wds else []
    refs += [{'ref_id': ref_id, 'split': ref['split'], 'category_id': ref['category_id'], 'ann_id': ref['ann_id'],
              'sent_ids': ref['sent_ids'], 'box': box, 'image_id': ref['image_id'],
              'att_wds': att_wds} ]
  print('There in all %s refs.' % len(refs))

  # prepare images = [{'image_id', 'width', 'height', 'file_name', 'ref_ids', 'ann_ids', 'h5_id'}]
  images = []
  h5_id = 0
  for image_id, image in refer.Imgs.items():
    width = image['width']
    height = image['height']
    file_name = image['file_name']
    ref_ids = [ref['ref_id'] for ref in refer.imgToRefs[image_id]]
    ann_ids = [ann['id'] for ann in refer.imgToAnns[image_id]]
    images += [ {'image_id': image_id, 'height': height, 'width': width, 'file_name': file_name, 'ref_ids': ref_ids, 'ann_ids': ann_ids, 'h5_id': h5_id} ]
    h5_id += 1
  print('There are in all %d images.' % h5_id)

  # prepare anns appeared in images, anns = [{ann_id, category_id, image_id, box, h5_id}]
  anns = []
  h5_id = 0
  for image_id in refer.Imgs:
    ann_ids = [ann['id'] for ann in refer.imgToAnns[image_id]]
    for ann_id in ann_ids:
      ann = refer.Anns[ann_id]
      anns += [{'ann_id': ann_id, 'category_id': ann['category_id'], 'box': ann['bbox'], 'image_id': image_id, 'h5_id': h5_id}]
      h5_id += 1
  print('There are in all %d anns within the %d images.' % (h5_id, len(images)))

  # prepare sentences = [{sent_id, tokens, h5_id, (dataset_splitBy)}]
  sentences = []
  h5_id = 0
  for sent_id, tokens in sentToFinal.items():
    sent = refer.Sents[sent_id]
    sent = {'sent_id': sent_id, 'tokens': tokens, 'h5_id': h5_id}
    if 'dataset_splitBy' in refer.Sents[sent_id]:
      sent['dataset_splitBy'] = refer.Sents[sent_id]['dataset_splitBy']
    sentences += [sent]
    # sentences += [{'sent_id': sent_id, 'tokens': tokens, 'h5_id': h5_id}]
    h5_id = h5_id + 1  
  print('There are in all %d sentences to written into hdf5 file.' % h5_id)

  return refs, images, anns, sentences

def build_att_vocab(refer, params, att_types=['r1', 'r2', 'r7']):
  """
  Load sents = [{tokens, atts, sent_id, parse, raw, sent left}] 
  from pyutils/refer-parser2/cache/parsed_atts/dataset_splitBy/sents.json
  """
  sents = json.load(open(osp.join('pyutils/refer-parser2/cache/parsed_atts', 
                                  params['dataset']+'_'+params['splitBy'], 'sents.json')))
  sentToRef = refer.sentToRef
  ref_to_att_wds = {}
  forbidden = forbidden_noun + forbidden_att + forbidden_verb \
              + list(refer.Cats.values()) # we also forbid category name here
  for sent in sents:
    sent_id = sent['sent_id']
    atts = sent['atts']
    ref_id = sentToRef[sent_id]['ref_id']
    for att_type in att_types:
      att_wds = [wd for wd in atts[att_type] if wd not in forbidden]
      if len(att_wds) > 0:
        ref_to_att_wds[ref_id] = ref_to_att_wds.get(ref_id, []) + att_wds
  ref_to_att_wds = {ref_id: list(set(att_wds)) for ref_id, att_wds in ref_to_att_wds.items()}

  # make vocab
  att2cnt = {}
  for ref_id, att_wds in ref_to_att_wds.items():
    for att_wd in att_wds:
      att2cnt[att_wd] = att2cnt.get(att_wd, 0) + 1
  sorted_att2cnt = sorted(att2cnt.items(), key=operator.itemgetter(1))[::-1]
  att2cnt = dict(sorted_att2cnt[:params['topK']])
  print('%s attribute words are chosen as vocabulary, which are mentioned %s times.' \
        % (len(att2cnt), sum(att2cnt.values())))

  # filter bad att_wds from ref_to_att_wds
  filtered_ref_to_att_wds = {}
  for ref_id, att_wds in ref_to_att_wds.items():
    att_wds = list(set(att_wds).intersection(set(att2cnt.keys())))
    if len(att_wds) > 0:
      filtered_ref_to_att_wds[ref_id] = att_wds
  print('%s refs have good attribute words.' % len(filtered_ref_to_att_wds))
  return att2cnt, filtered_ref_to_att_wds


def main(params):

  # dataset_splitBy
  data_root, dataset, splitBy = params['data_root'], params['dataset'], params['splitBy']

  # max_length
  if params['max_length'] == None:
    if params['dataset'] in ['refcoco', 'refclef', 'refcoco+']:
      params['max_length'] = 10
      params['topK'] = 50
    elif params['dataset'] in ['refcocog']:
      params['max_length'] = 20
      params['topK'] = 50
    else:
      raise NotImplementedError

  # mkdir and write json file
  if not osp.isdir(osp.join('cache/prepro', dataset+'_'+splitBy)):
    os.makedirs(osp.join('cache/prepro', dataset+'_'+splitBy))

  # load refer
  sys.path.insert(0, 'pyutils/refer')
  from refer import REFER
  refer = REFER(data_root, dataset, splitBy)

  # create vocab
  vocab, sentToFinal = build_vocab(refer, params)
  itow = {i: w for i, w in enumerate(vocab)} 
  wtoi = {w: i for i, w in enumerate(vocab)} 
  
  # check sentence length
  check_sentLength(sentToFinal)

  # create attribute vocab
  att2cnt, ref_to_att_wds = build_att_vocab(refer, params, ['r1','r2','r7']) 
  itoa = {i: a for i, a in enumerate(att2cnt.keys())}
  atoi = {a: i for i, a in enumerate(att2cnt.keys())}

  # prepare refs, images, anns, sentences
  # and write json
  refs, images, anns, sentences = prepare_json(refer, sentToFinal, ref_to_att_wds, params)
  json.dump({'refs': refs, 
             'images': images, 
             'anns': anns, 
             'sentences': sentences, 
             'word_to_ix': wtoi,
             'att_to_ix' : atoi,
             'att_to_cnt': att2cnt,
             'cat_to_ix': {cat_name: cat_id for cat_id, cat_name in refer.Cats.items()},
             'label_length': params['max_length'],}, 
             open(osp.join('cache/prepro', dataset+'_'+splitBy, params['output_json']), 'w'))
  print('%s written.' % osp.join('cache/prepro', params['output_json']))

  # write h5 file which contains /sentences
  f = h5py.File(osp.join('cache/prepro', dataset+'_'+splitBy, params['output_h5']), 'w')
  L = encode_captions(sentences, wtoi, params)
  f.create_dataset("labels", dtype='int32', data=L)
  f.close()
  print('%s writtern.' % osp.join('cache/prepro', params['output_h5']))

  # check
  # check_encoded_labels(sentences, L, itow)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--output_json', default='data.json', help='output json file')
  parser.add_argument('--output_h5', default='data.h5', help='output h5 file')
  parser.add_argument('--data_root', default='data', type=str, help='data folder containing images and four datasets.')
  parser.add_argument('--dataset', default='refcoco', type=str, help='refcoco/refcoco+/refcocog')
  parser.add_argument('--splitBy', default='unc', type=str, help='unc/google')
  parser.add_argument('--max_length', type=int, help='max length of a caption')  # refcoco 10, refclef 10, refcocog 20
  parser.add_argument('--images_root', default='', help='root location in which images are stored')
  parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
  parser.add_argument('--topK', default=50, type=int, help='top K attribute words')

  # argparse
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))

  # call main
  main(params)



