"""
8 diﬀerent target sequences of 7 diﬀerent subjects for training and testing

All videos are extracted at 60 frames per second (FPS)
and the synchronized audio waves are sampled at 16KHz frequency

alternatives:
(1) crop the video to keep the face at the center and then resize to cfg.audio_feature_size (prefer)
(2) directly resize to cfg.audio_feature_size

cfg.face_landmark_num pre-defned facial landmarks
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from datasets.face_dataset import FaceDataset
from tqdm import tqdm
from options.train_audio2feature_options import TrainOptions as FeatureOptions
from options.train_audio2headpose_options import TrainOptions as HeadposeOptions
from options.train_feature2face_options import TrainOptions as RenderOptions
from models.feature2face_model import Feature2FaceModel
import argparse
import config as cfg
from torch.utils.data import DataLoader
import copy
from os.path import join
import yaml
from models import create_model

def train():
    f_option = FeatureOptions()
    h_option = HeadposeOptions()
    r_option = RenderOptions()

    # --continue_train --load_epoch 0 --epoch_count 0
    #--load_pretrain xxx --debug --fp16 1 --local_rank 1 --verbose
    #--continue_train --TTUR --no_html
    #seq_max_len not use
    args_raw = f'--task Feature2Face --model feature2face --name Feature2Face --tf_log --gpu_ids 0\
        --dataset_mode face --dataset_names Vic --dataroot ./data \
        --isH5 1 --suffix .jpg --serial_batches --resize_or_crop scaleWidth  --no_flip 1 \
       --display_freq 100 --print_freq 10 --save_latest_freq 10 --save_epoch_freq 10 \
        --phase train --load_epoch latest --n_epochs_warm_up 5 \
        --n_epochs 100 --n_epochs_decay 100 --lr_decay_iters 1000 --lr_decay_gamma 0.25\
        --beta1 0.5 --lr 1e-4 --lr_final 1e-5 --lr_policy linear --gan_mode ls --pool_size 1 --frame_jump 1 \
        --epoch_count 0 --seq_max_len {cfg.FPS*2}'
    args_raw = args_raw.split(' ')
    args = []
    for x in args_raw:
        if len(x.strip()) > 0:
            args.append(x)
    f_option.isTrain = False
    h_option.isTrain = False
    r_option.isTrain = True
    opt = r_option.parse(args=args)

    dataset = FaceDataset(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=0,
                            drop_last=True)
    val_opt = copy.deepcopy(opt)
    val_opt.phase = 'val'
    val_dataset = FaceDataset(val_opt)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=0,
                                pin_memory=0,
                                drop_last=True)

    with open(join('./config_file/', opt.dataset_names[0] + '.yaml')) as f:
        config = yaml.load(f)
    data_root = opt.dataroot
    opt.size = config['model_params']['Image2Image']['size']

    print('---------- Create Model: {} -------------'.format(opt.task))
    #model = Feature2FaceModel()
    model = create_model(opt)
    model.setup(opt)

    for i_epoch in range(opt.n_epochs):
        iter_cnt = 0
        train_iter = iter(dataloader)
        print(f'i_epoch: {i_epoch}')
        while 1:
            try:
                batch = next(train_iter)
                iter_cnt += 1
                model.set_input(data=batch)
                model.optimize_parameters()
            except Exception as e:
                print(f'exception: {e}')
                print(f'iter_cnt: {iter_cnt}, loss_dict: {model.loss_dict}')
                break
        runned_epoch = i_epoch + 1
        if runned_epoch % opt.save_epoch_freq == 0:
            val_iter = iter(val_dataloader)
            while 1:
                try:
                    val_batch = next(val_iter)
                    model.set_input(data=batch)
                    model.validate()
                    print(f'val_batch loss_G: {model.loss_G}, loss_D: {model.loss_D}')
                except Exception as e:
                    #print(f'exception: {e}')
                    break
            # save mode
            model.save_networks(runned_epoch)


def main():
    train()


if __name__ == '__main__':
    main()
