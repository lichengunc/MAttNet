import os.path as osp
import sys 

# mrcn path
this_dir = osp.dirname(__file__)
mrcn_dir = osp.join(this_dir, '..', 'pyutils', 'mask-faster-rcnn')
sys.path.insert(0, osp.join(mrcn_dir, 'lib'))
sys.path.insert(0, osp.join(mrcn_dir, 'data', 'refer'))
sys.path.insert(0, osp.join(mrcn_dir, 'data', 'coco', 'PythonAPI'))

# refer path
refer_dir = osp.join(this_dir, '..', 'pyutils', 'refer')
sys.path.insert(0, refer_dir)

# model path
sys.path.insert(0, osp.join(this_dir, '..', 'lib'))