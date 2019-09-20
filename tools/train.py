import os
import os.path as osp
import sys
import numpy as np
import json
import h5py
import time
import random
from pprint import pprint

# dataset and model
import _init_paths
from loaders.matt_dataset import MAttDataset, matt_collate
from layers.joint_match import JointMatching
import models.utils as model_utils
import models.eval_easy_utils as eval_utils
from crits.max_margin_crit import MaxMarginCriterion
from opt import parse_opt

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# tensorboard
import tensorboardX as tb

def combine_feats(input_feats_list):
    output_feats = {}
    for k in input_feats_list[0].keys():
        output_feats[k] = torch.cat([feats[k] for feats in input_feats_list])
    return output_feats

# train one iter
def lossFun(data, model, mm_crit, optimizer, opt):
    # set mode
    model.train()

    # zero gradient
    optimizer.zero_grad()

    # time
    T = {}

    # fetch feats = {vfeats, lfeats, dif_lfeats, cxt_lfeats, cxt_vfeats}
    tic = time.time()
    ref_Feats = data['ref_Feats']  
    neg_Feats = data['neg_Feats']
    for _k in list(ref_Feats.keys()):
        # concat list of feats for each feat_type (key)
        ref_Feats[_k] = torch.cat([f.cuda() for f in ref_Feats[_k]], 0)
        neg_Feats[_k] = torch.cat([f.cuda() for f in neg_Feats[_k]], 0)
    
    # fetch labels
    ref_labels = torch.cat(data['ref_labels'], 0)
    neg_labels = torch.cat(data['neg_labels'], 0)
    max_len = max((ref_labels != 0).sum(1).max().item(), 
                  (neg_labels != 0).sum(1).max().item())
    ref_labels = ref_labels[:, :max_len].cuda()
    neg_labels = neg_labels[:, :max_len].cuda()
    T['feat'] = time.time() - tic

    # forward and backward model
    tic = time.time()
    Feats = combine_feats([ref_Feats, neg_Feats, ref_Feats])
    labels = torch.cat([ref_labels, ref_labels, neg_labels])
    scores, *_ = model(Feats['vfeats'], Feats['lfeats'], Feats['dif_lfeats'],
                       Feats['cxt_vfeats'], Feats['cxt_lfeats'], labels)
    loss = mm_crit(scores)
    loss.backward()
    model_utils.clip_gradient(optimizer, opt['grad_clip'])
    optimizer.step()
    T['model'] = time.time() - tic

    # return
    return loss.item(), T

def main(args):

    opt = vars(args)

    # initialize
    opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']
    checkpoint_dir = osp.join(opt['checkpoint_path'], 
                              opt['dataset_splitBy'], 
                              opt['id'])
    if not osp.isdir(checkpoint_dir): 
        os.makedirs(checkpoint_dir)

    # set random seed
    torch.manual_seed(opt['seed'])
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])

    # set up dataset
    data_json = osp.join('cache/prepro', opt['dataset_splitBy'], 'data.json')
    train_dataset = MAttDataset(data_json, 'train', opt['gd_feats_dir'], 
                        opt['seq_per_ref'], opt['visual_sample_ratio'],
                        opt['num_cxt'], opt['with_st'])
    val_dataset = MAttDataset(data_json, 'val', opt['gd_feats_dir'], 
                        opt['seq_per_ref'], opt['visual_sample_ratio'],
                        opt['num_cxt'], opt['with_st'])

    # set up loader
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=opt['batch_size'],
                              collate_fn=matt_collate,
                              num_workers=opt['num_workers'],
                              shuffle=True,
                              pin_memory=True)

    # set up model
    opt['fc7_dim'] = 2048
    opt['pool5_dim'] = 1024
    opt['vocab_size'] = train_dataset.vocab_size
    model = JointMatching(opt)

    # resume from previous checkpoint
    infos = {}
    if opt['start_from'] is not None:
        raise NotImplementedError
    iters = infos.get('iters', 0)
    epoch = infos.get('epoch', 0)
    val_accuracies = infos.get('accuracies', {})
    val_loss_history = infos.get('val_loss_history', {})
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    if opt['load_best_score'] == 1:
        best_val_score = infos.get('best_val_score', None)
    
    # set up criterion
    mm_crit = MaxMarginCriterion(opt['visual_rank_weight'], 
                                 opt['lang_rank_weight'], 
                                 opt['margin'])

    # move to GPU
    model.cuda()
    mm_crit.cuda()

    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=opt['learning_rate'],
                                 betas=(opt['optim_alpha'], opt['optim_beta']),
                                 eps=opt['optim_epsilon'])

    # tensorboard
    writer = tb.SummaryWriter(checkpoint_dir)

    # start training
    data_time, feat_time, model_time = 0, 0, 0
    lr = opt['learning_rate']
    losses_log_every = opt['losses_log_every']
    best_predictions, best_overall, best_epoch = None, None, None
    for epoch in range(opt['max_epochs']):
        # run one epoch
        tic_d = time.time()
        for i_batch, data in enumerate(train_loader):
            data_time += time.time() - tic_d

            # run batch forward/backward 
            loss, T = lossFun(data, model, mm_crit, optimizer, opt)
            feat_time += T['feat']
            model_time += T['model']
            iters += 1

            # write training loss summary
            if iters % losses_log_every == 0:
                loss_history[iters] = loss
                # print stats
                print(f'iters[{iters}](epoch[{epoch}]), '
                      f'train_loss={loss:.3f}, '
                      f'lr={lr:.2E}, '
                      f'data:{data_time/losses_log_every:.2f}s/iter, '
                      f'feat:{feat_time/losses_log_every:.2f}s/iter, '
                      f'model:{model_time/losses_log_every:.2f}s/iter.')
                data_time, feat_time, model_time = 0, 0, 0
                # tb log
                writer.add_scalar('train/loss', loss, iters)
                writer.add_scalar('train/lr', lr, iters)

            # decay the learning rates
            if opt['learning_rate_decay_start'] > 0 and \
                    iters > opt['learning_rate_decay_start']:
                frac = (iters - opt['learning_rate_decay_start']) / \
                        opt['learning_rate_decay_every']
                decay_factor =  0.1 ** frac
                lr = opt['learning_rate'] * decay_factor
                # update optimizer's learning rate
                model_utils.set_lr(optimizer, lr)

            # next data timer
            tic_d = time.time()

        # evaluate loss and save checkpoint
        val_acc, predictions = eval_utils.eval_split(
                val_dataset, model, opt)
        val_accuracies[iters] = val_acc
        print('validation acc : %.2f%%\n' % (val_acc*100.0))
        # tb log
        writer.add_scalar('val/accuracy', val_acc, iters)

        # save model if best
        current_score = val_acc
        if best_val_score is None or current_score > best_val_score:
            best_val_score = current_score
            best_predictions = predictions
            best_epoch = epoch
            checkpoint_path = osp.join(checkpoint_dir, 'model.pth')
            checkpoint = {}
            # checkpoint['model'] = model 
            checkpoint['model'] = model.state_dict()  
            checkpoint['opt'] = opt
            torch.save(checkpoint, checkpoint_path) 
            print('model saved to %s' % checkpoint_path) 

        # write json report
        infos['iters'] = iters
        infos['epoch'] = epoch
        infos['loss_history'] = loss_history
        infos['val_accuracies'] = val_accuracies
        infos['best_val_score'] = best_val_score
        infos['best_epoch'] = best_epoch
        infos['best_predictions'] = best_predictions
        infos['opt'] = opt
        infos['word_to_ix'] = train_dataset.word_to_ix
        with open(osp.join(checkpoint_dir, 'report.json'), 'w') as io:
            json.dump(infos, io)


if __name__ == '__main__':

  args = parse_opt()
  main(args)

