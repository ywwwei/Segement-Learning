# ------------------------------------------------------------------------
# SeqFormer data loader
# ------------------------------------------------------------------------
# Modified from Deformable VisTR (https://github.com/Epiphqny/VisTR)
# ------------------------------------------------------------------------


from pathlib import Path

import torch
import torch.utils.data
from pycocotools.ytvos import YTVOS
import datasets.transform_clip as T
import os
from PIL import Image
from random import randint
import cv2
import random
import math

import math
import time


class YTVOSDataset:
    def __init__(self, img_folder, ann_file, transforms, return_masks, num_frames):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks
        self.num_frames = num_frames
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.vid_ids = self.ytvos.getVidIds()
        self.vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            self.vid_infos.append(info)
        self.img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                self.img_ids.append((idx, frame_id))

        print('\n video num:', len(self.vid_ids), '  clip num:', len(self.img_ids)) 
        print('\n')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        instance_check = False
        while not instance_check:          
            vid,  frame_id = self.img_ids[idx]
            vid_id = self.vid_infos[vid]['id']
            img = []
            vid_len = len(self.vid_infos[vid]['file_names'])
            inds = list(range(self.num_frames))
            num_frames = self.num_frames
            # random sparse sample
            sample_indx = [frame_id]
            #local sample
            samp_id_befor = randint(1,3)
            samp_id_after = randint(1,3)
            local_indx = [max(0, frame_id - samp_id_befor), min(vid_len - 1, frame_id + samp_id_after)]
            sample_indx.extend(local_indx)

            # global sampling
            if num_frames > 3:
                all_inds = list(range(vid_len))
                global_inds = all_inds[:min(sample_indx)]+all_inds[max(sample_indx):]
                global_n = num_frames - len(sample_indx)
                if len(global_inds) > global_n:
                    select_id = random.sample(range(len(global_inds)),global_n)
                    for s_id in select_id:
                        sample_indx.append(global_inds[s_id])
                elif vid_len >=global_n:  # sample long range global frames
                    select_id = random.sample(range(vid_len),global_n)
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                else:
                    select_id = random.sample(range(vid_len),global_n - vid_len)+list(range(vid_len))           
                    for s_id in select_id:                                                                   
                        sample_indx.append(all_inds[s_id])
            sample_indx.sort()
    
            for j in range(self.num_frames):
                img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][sample_indx[j]])
                img.append(Image.open(img_path).convert('RGB'))

            if self._transforms is not None:
                img, _ = self._transforms(img, target=None, num_frames=num_frames)

        return torch.cat(img,dim=0), None

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # scales = [296, 328, 360, 392]
    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=512),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([300, 400, 500]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=512),
                    T.Check(),
                ])
            ),
            normalize,
        ])


    if image_set == 'val':
        return T.Compose([
            # T.RandomResize([800], max_size=1333),
            T.RandomResize([360], max_size=640),
            normalize,
        ])
        
    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.ytvis_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    
    if args.dataset_file == 'YoutubeVIS' or args.dataset_file == 'jointcoco':
        # mode = 'instances'
        PATHS = {
            "train": (root / "train/JPEGImages", root / "annotations" / f'train.v4.json'),
            "val": (root / "train/JPEGImages", root / "annotations" / f'valid.v4.json'),
        }
        img_folder, ann_file = PATHS[image_set]
        print('use Youtube-VIS dataset')
        dataset = YTVOSDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,  num_frames = args.num_frames)

    return dataset
