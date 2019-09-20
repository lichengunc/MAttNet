"""
data_json has 
0. refs:[{ref_id, ann_id, box, image_id, split, category_id, sent_ids, att_wds}]
1. images:     [{image_id, ref_ids, ann_ids, file_name, width, height, h5_id}]
2. anns:       [{ann_id, category_id, image_id, box, h5_id}]
3. sentences:  [{sent_id, tokens, h5_id}]
4. word_to_ix: {word: ix}
5. att_to_ix : {att_wd: ix}
6. att_to_cnt: {att_wd: cnt}
7. label_length: L

Note, box in [xywh] format
label_h5 has
/labels is (M, max_length) uint32 array of encoded labels, zeros padded in back
"""
import os.path as osp
import numpy as np
import h5py
import json
import random
import collections
import functools

import torch
import torch.utils.data as Data

def matt_collate(batch):
    """
    Used to collate matt_dataset into batch.
    """
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    if torch.is_tensor(batch[0]):
        # collate feats and labels
        return [_ for _ in batch]
    elif isinstance(batch[0], collections.Sequence):
        # collate list
        return [_ for _ in batch]
    elif isinstance(batch[0], int):
        # collate image_id
        return [_ for _ in batch]
    elif isinstance(batch[0], collections.Mapping):
        return {key: matt_collate([d[key] for d in batch]) for key in batch[0]}
    raise TypeError((error_msg.format(type(batch[0]))))

class MAttDataset(Data.Dataset):

    def __init__(self, data_json, split, feats_dir, 
                 seq_per_ref=3, visual_sample_ratio=0.3, 
                 num_cxt=5, with_st=1):
        # Load data
        print('Dataset loading data.json: ', data_json)
        self.info = json.load(open(data_json))
        self.word_to_ix = self.info['word_to_ix'] 
        self.ix_to_word = {ix: wd for wd, ix in self.word_to_ix.items()}
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)
        self.cat_to_ix = self.info['cat_to_ix']
        self.ix_to_cat = {ix: cat for cat, ix in self.cat_to_ix.items()}
        print('object category size is ', len(self.ix_to_cat))
        self.images = self.info['images']
        self.anns = self.info['anns']
        self.refs = self.info['refs']
        self.sentences = self.info['sentences']
        print(f'We have {len(self.images)} images.')
        print(f'We have {len(self.anns)} anns.')
        print(f'We have {len(self.refs)} refs.')
        print(f'We have {len(self.sentences)} sentences.')
        self.label_length = self.info['label_length']
        print(f'label_length is {self.label_length}.')
        
        # construct mappings
        self.Refs = {ref['ref_id']: ref for ref in self.refs}
        self.Images = {image['image_id']: image for image in self.images}
        self.Anns = {ann['ann_id']: ann for ann in self.anns}
        self.Sentences = {sent['sent_id']: sent for sent in self.sentences}
        self.annToRef = {ref['ann_id']: ref for ref in self.refs}
        self.sentToRef = {sent_id: ref for ref in self.refs 
                            for sent_id in ref['sent_ids']}

        # prepare attributes
        self.att_to_ix = self.info['att_to_ix']
        self.ix_to_att = {ix: wd for wd, ix in self.att_to_ix.items()}
        self.att_to_cnt = self.info['att_to_cnt']
        self.attribute_size = len(self.att_to_ix)

        # image_ids of each split
        self.split = split
        self.split_image_ids = []
        for image_id, image in self.Images.items():
            if self.Refs[image['ref_ids'][0]]['split'] == self.split:
                self.split_image_ids += [image_id]
        print(f'Assigned {len(self.split_image_ids)} images to [{self.split}].')

        # other options
        self.seq_per_ref = seq_per_ref
        self.sample_ratio = visual_sample_ratio
        self.num_cxt = num_cxt
        self.with_st = with_st
        self.feats_dir = feats_dir
    
    def __len__(self):
        return len(self.split_image_ids)
    
    def __getitem__(self, index):
        """
        Inputs:
        :index: indexing image_id
        Returns:
        :image_id
        :image_ann_ids
        :ref_ids
        :ref_ann_ids
        :ref_sent_ids
        :ref_labels      : (n, label_length) long
        :ref_cxt_ann_ids : (n, num_cxt) ann_ids for each of ref_id, -1 padded
        :ref_Feats       : vfeats     (n, 2048) float
                         : lfeats     (n, 5) float
                         : dif_lfeats (n, 25) float
                         : cxt_lfeats (n, num_cxt, 5) float
        :neg_ann_ids     : n negative anns 
        :neg_sent_ids    : n negative sent_ids
        :neg_labels      : (n, label_length) long
        :neg_cxt_ann_ids : (n, num_cxt) ann_ids for each of neg_id, -1 padded
        :neg_Feats       : vfeats     (n, 2048) float
                         : lfeats     (n, 5) float
                         : dif_lfeats (n, 25) float
                         : cxt_lfeats (n, num_cxt, 5) float
        """
        # get image and its feats
        image_id = self.split_image_ids[index]
        image_ann_ids = self.Images[image_id]['ann_ids'] # ordered w.r.t feats
        image_ann_feats = self.get_image_feats(image_id) # (#ann_ids, k)

        # expand ref_dis by seq_per_ref
        ref_ids = self.Images[image_id]['ref_ids']
        image_ref_ids = self.expand_list(ref_ids, self.seq_per_ref)

        # sample all ids
        ref_ann_ids, ref_sent_ids = [], []
        neg_ann_ids, neg_sent_ids = [], []
        for ref_id in ref_ids:
            ref_ann_id = self.Refs[ref_id]['ann_id']

            # pos ids
            ref_ann_ids += [ref_ann_id] * self.seq_per_ref
            ref_sent_ids += self.fetch_sent_ids_by_ref_id(
                                ref_id, self.seq_per_ref)

            # neg ids
            cur_ann_ids, cur_sent_ids = self.sample_neg_ids(
                ref_ann_id, self.seq_per_ref, self.sample_ratio)
            neg_ann_ids += cur_ann_ids
            neg_sent_ids += cur_sent_ids
        
        # compute all lfeats
        ref_vfeats = self.extract_ann_feats(image_ann_feats, 
                            image_ann_ids, ref_ann_ids)
        ref_lfeats = self.compute_lfeats(ref_ann_ids)
        ref_dif_lfeats = self.compute_dif_lfeats(ref_ann_ids)
        neg_vfeats = self.extract_ann_feats(image_ann_feats, 
                            image_ann_ids, neg_ann_ids)
        neg_lfeats = self.compute_lfeats(neg_ann_ids)
        neg_dif_lfeats = self.compute_dif_lfeats(neg_ann_ids)

        # fetch labels
        ref_labels = self.fetch_labels(ref_sent_ids)
        neg_labels = self.fetch_labels(neg_sent_ids)

        # fetch context info: cxt_lfeats, cxt_ann_ids
        ref_cxt_vfeats, ref_cxt_lfeats, ref_cxt_ann_ids = self.extract_context(
            ref_ann_ids, self.num_cxt, image_ann_feats, image_ann_ids)
        neg_cxt_vfeats, neg_cxt_lfeats, neg_cxt_ann_ids = self.extract_context(
            neg_ann_ids, self.num_cxt, image_ann_feats, image_ann_ids)

        # return
        data = {}
        data['image_id'] = image_id
        data['image_ann_ids'] = image_ann_ids
        data['ref_ids'] = image_ref_ids
        data['ref_ann_ids'] = ref_ann_ids
        data['ref_sent_ids'] = ref_sent_ids
        data['ref_labels'] = torch.from_numpy(ref_labels).long()
        data['ref_cxt_ann_ids'] = ref_cxt_ann_ids
        data['ref_Feats'] = {
            'vfeats': torch.from_numpy(ref_vfeats).float(), 
            'lfeats': torch.from_numpy(ref_lfeats).float(),
            'dif_lfeats': torch.from_numpy(ref_dif_lfeats).float(), 
            'cxt_vfeats': torch.from_numpy(ref_cxt_vfeats).float(),
            'cxt_lfeats': torch.from_numpy(ref_cxt_lfeats).float()}
        data['neg_ann_ids'] = neg_ann_ids
        data['neg_sent_ids'] = neg_sent_ids
        data['neg_labels'] = torch.from_numpy(neg_labels).long()
        data['neg_cxt_ann_ids'] = neg_cxt_ann_ids
        data['neg_Feats'] = {
            'vfeats': torch.from_numpy(neg_vfeats).float(), 
            'lfeats': torch.from_numpy(neg_lfeats).float(), 
            'dif_lfeats': torch.from_numpy(neg_dif_lfeats).float(), 
            'cxt_vfeats': torch.from_numpy(neg_cxt_vfeats).float(),
            'cxt_lfeats': torch.from_numpy(neg_cxt_lfeats).float()}
        return data

    def get_image_feats(self, image_id):
        """return (n, 2048) ann_feats given image_id"""
        file_name = f'visual_grounding_coco_gt_{image_id:012}.npz'
        img_dump = np.load(f'{self.feats_dir}/{file_name}', allow_pickle=True)
        return img_dump['features']
    
    def expand_list(self, L, n):
        """e.g., [a, b], 3 --> [aaabbb]"""
        out = []
        for l in L:
            out += [l] * n
        return out
    
    def extract_ann_feats(self, image_ann_feats, image_ann_ids, tgt_ann_ids):
        """ 
        Extract ann_feats according to tgt_ann_ids within image_ann_ids
        Note there might be -1 in tgt_ann_ids, its feats is zeros.
        """
        feats = np.zeros((len(tgt_ann_ids), image_ann_feats.shape[1]))
        for i, tgt_ann_id in enumerate(tgt_ann_ids):
            if tgt_ann_id != -1:
                # tgt_ann_id in image_ann_ids:  # rare tgt_ann_id cannot be found
                feats[i] = image_ann_feats[image_ann_ids.index(tgt_ann_id)]
        return feats

    def fetch_sent_ids_by_ref_id(self, ref_id, num_sents):
        """
        Sample #num_sents sents for each ref_id.
        """
        sent_ids = list(self.Refs[ref_id]['sent_ids'])
        if len(sent_ids) < num_sents:
            append_sent_ids = [random.choice(sent_ids) 
                                for _ in range(num_sents - len(sent_ids))]
            sent_ids += append_sent_ids
        else:
            random.shuffle(sent_ids)
            sent_ids = sent_ids[:num_sents]
        assert len(sent_ids) == num_sents
        return sent_ids

    def sample_neg_ids(self, ann_id, seq_per_ref, sample_ratio):
        """Return
        - neg_ann_ids : list of ann_ids that are negative to target ann_id
        - neg_sent_ids: list of sent_ids that are negative to target ann_id
        """
        st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids = \
            self.fetch_neighbour_ids(ann_id)
        # neg ann
        neg_ann_ids, neg_sent_ids = [], []
        for k in range(seq_per_ref):
            # neg_ann_id for negative visual representation: 
            # mainly from same-type objects
            if len(st_ann_ids) > 0 and np.random.uniform(0,1,1) < sample_ratio:
                neg_ann_id = random.choice(st_ann_ids)
            elif len(dt_ann_ids) > 0:
                neg_ann_id = random.choice(dt_ann_ids)
            else:
                # awkward case: I just randomly sample 
                # from st_ann_ids + dt_ann_ids, or -1
                if len(st_ann_ids + dt_ann_ids) > 0:
                    neg_ann_id = random.choice(st_ann_ids + dt_ann_ids)
                else:
                    neg_ann_id = -1
            neg_ann_ids += [neg_ann_id]
            # neg_ref_id for negative language representations: 
            # mainly from same-type "referred" objects
            if len(st_ref_ids) > 0 and np.random.uniform(0,1,1) < sample_ratio:
                neg_ref_id = random.choice(st_ref_ids)
            elif len(dt_ref_ids) > 0:
                neg_ref_id = random.choice(dt_ref_ids)
            else:
                neg_ref_id = random.choice(list(self.Refs.keys()))
            neg_sent_id = random.choice(self.Refs[neg_ref_id]['sent_ids'])
            neg_sent_ids += [neg_sent_id]
        # return
        return neg_ann_ids, neg_sent_ids
    
    def fetch_neighbour_ids(self, ref_ann_id):
        """
        For a given ref_ann_id, we return
        - st_ann_ids: same-type neighbouring ann_ids (not including itself)
        - dt_ann_ids: different-type neighbouring ann_ids
        Ordered by distance to the input ann_id
        """
        ref_ann = self.Anns[ref_ann_id]
        x, y, w, h = ref_ann['box']
        rx, ry = x+w/2, y+h/2
        def my_compare(ann_id0, ann_id1):
            x, y, w, h = self.Anns[ann_id0]['box']
            ax0, ay0 = x+w/2, y+h/2
            x, y, w, h = self.Anns[ann_id1]['box']
            ax1, ay1 = x+w/2, y+h/2
            # closer --> former
            if (rx-ax0)**2 + (ry-ay0)**2 <= (rx-ax1)**2 + (ry-ay1)**2:
                return -1
            else:
                return 1
        image = self.Images[ref_ann['image_id']]   
        ann_ids = list(image['ann_ids'])  # copy in case the raw list is changed
        ann_ids = sorted(ann_ids, key=functools.cmp_to_key(my_compare))

        st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids = [], [], [], []
        for ann_id in ann_ids:
            if ann_id != ref_ann_id:
                if self.Anns[ann_id]['category_id'] == ref_ann['category_id']:
                    st_ann_ids += [ann_id]
                    if ann_id in self.annToRef:
                        st_ref_ids += [self.annToRef[ann_id]['ref_id']]
                else:
                    dt_ann_ids += [ann_id]
                    if ann_id in self.annToRef:
                        dt_ref_ids += [self.annToRef[ann_id]['ref_id']]
        return st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids

    def extract_context(self, ann_ids, topK, image_ann_feats, image_ann_ids):
        """
        Return:
        :cxt_vfeats : ndarray float32 (#ann_ids, topK, 2048), padded with 0
        :cxt_lfeats : ndarray float32 (#ann_ids, topK, 5), padded with 0
        :cxt_ann_ids: list[[ann_id]] of size (#ann_ids, topK), padded with -1
        Note we only use neighbouring "different"(+"same") objects for 
        computing context objects.
        """
        cxt_vfeats = np.zeros((len(ann_ids), topK, image_ann_feats.shape[1]))
        cxt_lfeats = np.zeros((len(ann_ids), topK, 5), dtype=np.float32)
        cxt_ann_ids = [[-1 for _ in range(topK)] 
                        for _ in range(len(ann_ids))] # (#ann_ids, topK)
        for i, ref_ann_id in enumerate(ann_ids):
            if ref_ann_id == -1:
                continue
            # reference box
            rbox = self.Anns[ref_ann_id]['box']
            rcx, rcy = rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2
            rw, rh = rbox[2], rbox[3]
            rw += 1e-5
            rh += 1e-5
            # candidate boxes
            _, st_ann_ids, _, dt_ann_ids = self.fetch_neighbour_ids(ref_ann_id)
            if self.with_st > 0:
                cand_ann_ids = dt_ann_ids + st_ann_ids
            else:
                cand_ann_ids = dt_ann_ids
            cand_ann_ids = cand_ann_ids[:topK]
            for j, cand_ann_id in enumerate(cand_ann_ids):
                cand_ann = self.Anns[cand_ann_id]
                cbox = cand_ann['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                cxt_lfeats[i, j, :] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, \
                        (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
                cxt_ann_ids[i][j] = cand_ann_id
                cxt_vfeats[i][j] = image_ann_feats[
                                    image_ann_ids.index(cand_ann_id)]
        return cxt_vfeats, cxt_lfeats, cxt_ann_ids

    def compute_lfeats(self, ann_ids):
        # return ndarray float32 (#ann_ids, 5)
        lfeats = np.zeros((len(ann_ids), 5), dtype=np.float32)
        for ix, ann_id in enumerate(ann_ids):
            if ann_id == -1:
                continue
            ann = self.Anns[ann_id]
            image = self.Images[ann['image_id']]
            x, y, w, h = ann['box']
            ih, iw = image['height'], image['width']
            lfeats[ix] = np.array([[x/iw, y/ih, (x+w-1)/iw, 
                            (y+h-1)/ih, w*h/(iw*ih)]], np.float32) 
        return lfeats
    
    def compute_dif_lfeats(self, ann_ids, topK=5):
        # return ndarray float32 (#ann_ids, 5*topK)
        dif_lfeats = np.zeros((len(ann_ids), 5*topK), dtype=np.float32)
        for i, ref_ann_id in enumerate(ann_ids):
            if ref_ann_id == -1:
                continue
            # reference box
            rbox = self.Anns[ref_ann_id]['box']
            rcx, rcy = rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2
            rw, rh = rbox[2], rbox[3]
            rw += 1e-5
            rh += 1e-5
            # candidate boxes
            _, st_ann_ids, _, _ = self.fetch_neighbour_ids(ref_ann_id)
            for j, cand_ann_id in enumerate(st_ann_ids[:topK]):
                cbox = self.Anns[cand_ann_id]['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                dif_lfeats[i, j*5:(j+1)*5] = np.array([(cx1-rcx)/rw, 
                 (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
        return dif_lfeats

    def encode_labels(self, tokens_list):
        # encode to np.int64 (num_tokens_list, label_length)
        L = np.zeros((len(tokens_list), self.label_length), dtype=np.int64)
        for i, tokens in enumerate(tokens_list):
            for j, w in enumerate(tokens):
                if j < self.label_length:
                    L[i, j] = self.word_to_ix[w]
        return L

    def fetch_labels(self, sent_ids):
        """
        Return: int64 (num_sents, label_length)
        """
        tokens_list = [self.Sentences[sent_id]['tokens'] 
                        for sent_id in sent_ids]
        labels = self.encode_labels(tokens_list)
        return labels

    def decode_labels(self, labels):
        """
        labels: int32 (n, label_length) zeros padded in end
        return: list of sents in string format
        """
        decoded_sent_strs = []
        num_sents = labels.shape[0]
        for i in range(num_sents):
            label = labels[i].tolist()
            sent_str = ' '.join([self.ix_to_word[int(i)] 
                        for i in label if i != 0])
            decoded_sent_strs.append(sent_str)
        return decoded_sent_strs
    
    def getImageBatch(self, image_id, sent_ids=None):
        """
        Inputs:
        :image_id : int
        Returns:
        :image_id       : same as input
        :ann_ids        : N annotated objects in the image
        :cxt_ann_ids    : float (N, num_cxt)
        :sent_ids       : M sent_ids used in this image
        :Feats          : - lfeats     float (N, 5)
                          - dif_lfeats float (N, 25)
                          - cxt_lfeats float (N, num_cxt, 5)
        :labels         : long (M, label_length)
        :gd_boxes       : list of M [xywh]
        :gd_ann_ids     : list of M ann_ids
        :ref_ids        : list of M ref_ids
        """
        image = self.Images[image_id]
        image_ann_ids = self.Images[image_id]['ann_ids'] # ordered w.r.t feats
        image_ann_feats = self.get_image_feats(image_id) # (#ann_ids, k)

        # compute all lfeats
        vfeats = image_ann_feats 
        lfeats = self.compute_lfeats(image_ann_ids)
        dif_lfeats = self.compute_dif_lfeats(image_ann_ids)

        # get context
        cxt_vfeats, cxt_lfeats, cxt_ann_ids = self.extract_context(
            image_ann_ids, self.num_cxt, image_ann_feats, image_ann_ids)

        # fetch sent_ids and labels
        gd_ann_ids, gd_boxes = [], []
        if sent_ids is None:
            sent_ids = []
            for ref_id in image['ref_ids']:
                ref = self.Refs[ref_id]
                for sent_id in ref['sent_ids']:
                    sent_ids += [sent_id]
                    gd_ann_ids += [ref['ann_id']]
                    gd_boxes += [ref['box']]  # xywh
        else:
            # given sent_id, we find the gd_ix
            for sent_id in sent_ids:
                ref = self.sentToRef[sent_id]
                gd_ann_ids += [ref['ann_id']]
                gd_boxes += [ref['box']]
        labels = self.fetch_labels(sent_ids)

        # return data
        data = {}
        data['image_id'] = image_id
        data['ann_ids'] = image_ann_ids
        data['cxt_ann_ids'] = cxt_ann_ids
        data['sent_ids'] = sent_ids
        data['gd_ann_ids'] = gd_ann_ids
        data['gd_boxes'] = gd_boxes
        data['Feats'] = {'vfeats': torch.from_numpy(vfeats).float(),
                         'lfeats': torch.from_numpy(lfeats).float(),
                         'dif_lfeats': torch.from_numpy(dif_lfeats).float(),
                         'cxt_vfeats': torch.from_numpy(cxt_vfeats).float(),
                         'cxt_lfeats': torch.from_numpy(cxt_lfeats).float()}
        data['labels'] = torch.from_numpy(labels).long()
        return data


class MAttDetectionDataset(MAttDataset):

    def __init__(self, data_json, dets_json, iid_to_det_ids_json, split, 
                 feats_dir, seq_per_ref=3, visual_sample_ratio=0.3, 
                 num_cxt=5, with_st=1):
        # Load Dets
        self.dets = json.load(open(dets_json))
        self.Dets = {det['det_id']: det for det in self.dets}

        # Load data
        print('Dataset loading data.json: ', data_json)
        self.info = json.load(open(data_json))
        self.word_to_ix = self.info['word_to_ix'] 
        self.ix_to_word = {ix: wd for wd, ix in self.word_to_ix.items()}
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)
        self.cat_to_ix = self.info['cat_to_ix']
        self.ix_to_cat = {ix: cat for cat, ix in self.cat_to_ix.items()}
        print('object category size is ', len(self.ix_to_cat))
        self.images = self.info['images']
        self.anns = self.info['anns']
        self.refs = self.info['refs']
        self.sentences = self.info['sentences']
        print(f'We have {len(self.images)} images.')
        print(f'We have {len(self.anns)} anns.')
        print(f'We have {len(self.refs)} refs.')
        print(f'We have {len(self.sentences)} sentences.')
        self.label_length = self.info['label_length']
        print(f'label_length is {self.label_length}.')

        # add dets to image
        iid_to_det_ids = json.load(
            open(iid_to_det_ids_json, 'r'))['iid_to_det_ids']
        iid_to_det_ids = {int(iid): det_ids 
            for iid, det_ids in iid_to_det_ids.items()}
        for image in self.images:
            if image['image_id'] in iid_to_det_ids:
                image['det_ids'] = iid_to_det_ids[image['image_id']]

        # construct mappings
        self.Refs = {ref['ref_id']: ref for ref in self.refs}
        self.Images = {image['image_id']: image for image in self.images}
        self.Anns = {ann['ann_id']: ann for ann in self.anns}
        self.Sentences = {sent['sent_id']: sent for sent in self.sentences}
        self.annToRef = {ref['ann_id']: ref for ref in self.refs}
        self.sentToRef = {sent_id: ref for ref in self.refs 
                            for sent_id in ref['sent_ids']}
        
        # image_ids of each split
        self.split = split
        self.split_image_ids = []
        for image_id, image in self.Images.items():
            if self.Refs[image['ref_ids'][0]]['split'] == self.split:
                self.split_image_ids += [image_id]
        print(f'Assigned {len(self.split_image_ids)} images to [{self.split}].')

        # other options
        self.seq_per_ref = seq_per_ref
        self.sample_ratio = visual_sample_ratio
        self.num_cxt = num_cxt
        self.with_st = with_st
        self.feats_dir = feats_dir

    def fetch_neighbour_ids(self, ref_det_id):
        """
        For a given ref_det_id, we return
        - st_det_ids: same-type neighbouring det_ids (not including itself)
        - dt_det_ids: different-type neighbouring det_ids
        Ordered by distance to the input det_id
        """
        ref_det = self.Dets[ref_det_id]
        x, y, w, h = ref_det['box']
        rx, ry = x+w/2, y+h/2

        def my_compare(det_id0, det_id1):
            x, y, w, h = self.Dets[det_id0]['box']
            ax0, ay0 = x+w/2, y+h/2
            x, y, w, h = self.Dets[det_id1]['box']
            ax1, ay1 = x+w/2, y+h/2
            # closer --> former
            if (rx-ax0)**2 + (ry-ay0)**2 <= (rx-ax1)**2 + (ry-ay1)**2:
                return -1
            else:
                return 1
        image = self.Images[ref_det['image_id']]

        det_ids = list(image['det_ids'])  # copy in case the raw list is changed
        det_ids = sorted(det_ids, key=functools.cmp_to_key(my_compare))

        st_det_ids, dt_det_ids = [], []
        for det_id in det_ids:
            if det_id != ref_det_id:
                if self.Dets[det_id]['category_id'] == ref_det['category_id']:
                    st_det_ids += [det_id]
                else:
                    dt_det_ids += [det_id]

        return st_det_ids, dt_det_ids    

    def get_image_feats(self, image_id):
        """return (n, 2048) feats"""
        file_name = f'visual_grounding_det_coco_{image_id:012}.npz'
        img_dump = np.load(f'{self.feats_dir}/{file_name}', allow_pickle=True)
        return img_dump['features'] 

    def compute_lfeats(self, det_ids):
        # return ndarray float32 (#det_ids, 5)
        lfeats = np.empty((len(det_ids), 5), dtype=np.float32)
        for ix, det_id in enumerate(det_ids):
            det = self.Dets[det_id]
            image = self.Images[det['image_id']]
            x, y, w, h = det['box']
            ih, iw = image['height'], image['width']
            lfeats[ix] = np.array([[x/iw, y/ih, 
                    (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)]], np.float32) 
        return lfeats

    def compute_dif_lfeats(self, det_ids, topK=5):
        # return ndarray float32 (#det_ids, 5*topK)
        dif_lfeats = np.zeros((len(det_ids), 5*topK), dtype=np.float32)
        for i, ref_det_id in enumerate(det_ids):
            # reference box 
            rbox = self.Dets[ref_det_id]['box']
            rcx, rcy, rw, rh = (rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2, 
                                rbox[2], rbox[3])
            # candidate boxes
            st_det_ids, dt_det_ids = self.fetch_neighbour_ids(ref_det_id)
            for j, cand_det_id in enumerate(st_det_ids[:topK]):
                cbox = self.Dets[cand_det_id]['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                dif_lfeats[i, j*5:(j+1)*5] = \
                    np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, 
                              (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
        return dif_lfeats

    def extract_context(self, det_ids, topK, image_det_feats, image_det_ids):
        """
        Return:
        :cxt_vfeats : ndarray float32 (#det_ids, topK, 2048), padded with 0
        :cxt_lfeats : ndarray float32 (#det_ids, topK, 5), padded with 0
        :cxt_det_ids: list[[det_id]] of size (#det_ids, topK), padded with -1
        Note we only use neighbouring "different"(+"same") objects for 
        computing context objects.
        """
        cxt_vfeats = np.zeros((len(det_ids), topK, image_det_feats.shape[1]))
        cxt_lfeats = np.zeros((len(det_ids), topK, 5), dtype=np.float32)
        cxt_det_ids = [[-1 for _ in range(topK)] 
                        for _ in range(len(det_ids))] # (#det_ids, topK)
        for i, ref_det_id in enumerate(det_ids):
            # reference box
            rbox = self.Dets[ref_det_id]['box']
            rcx, rcy, rw, rh = (rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2, 
                                rbox[2], rbox[3])
            rw += 1e-5
            rh += 1e-5
            # candidate boxes
            st_det_ids, dt_det_ids = self.fetch_neighbour_ids(ref_det_id)
            if self.with_st > 0:
                cand_det_ids = dt_det_ids + st_det_ids
            else:
                cand_det_ids = dt_det_ids
            cand_det_ids = cand_det_ids[:topK]
            for j, cand_det_id in enumerate(cand_det_ids):
                cand_det = self.Dets[cand_det_id]
                cbox = cand_det['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                cxt_lfeats[i, j, :] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, 
                            (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
                cxt_det_ids[i][j] = cand_det_id
                cxt_vfeats[i][j] = image_det_feats[
                                    image_det_ids.index(cand_det_id)]
        return cxt_vfeats, cxt_lfeats, cxt_det_ids

    def getImageBatch(self, image_id):
        """
        Inputs:
        :image_id : int
        : 
        """
        image = self.Images[image_id]
        image_det_ids = image['det_ids']  # ordered w.r.t feats
        image_det_feats = self.get_image_feats(image_id)

        # compute all lfeats
        vfeats = image_det_feats
        lfeats = self.compute_lfeats(image_det_ids)
        dif_lfeats = self.compute_dif_lfeats(image_det_ids)

        # get context
        cxt_vfeats, cxt_lfeats, cxt_det_ids = self.extract_context(
            image_det_ids, self.num_cxt, image_det_feats, image_det_ids)
        
        # fetch sent_ids and labels
        sent_ids = []
        gd_boxes = []
        for ref_id in image['ref_ids']:
            ref = self.Refs[ref_id]
            for sent_id in ref['sent_ids']:
                sent_ids.append(sent_id)
                gd_boxes.append(ref['box'])  # xywh
        labels = self.fetch_labels(sent_ids)

        # return data
        data = {}
        data['image_id'] = image_id
        data['det_ids'] = image_det_ids
        data['cxt_det_ids'] = cxt_det_ids
        data['sent_ids'] = sent_ids
        data['gd_boxes'] = gd_boxes
        data['Feats'] = {'vfeats': torch.from_numpy(vfeats).float(),
                         'lfeats': torch.from_numpy(lfeats).float(),
                         'dif_lfeats': torch.from_numpy(dif_lfeats).float(),
                         'cxt_vfeats': torch.from_numpy(cxt_vfeats).float(),
                         'cxt_lfeats': torch.from_numpy(cxt_lfeats).float()}
        data['labels'] = torch.from_numpy(labels).long()
        return data


