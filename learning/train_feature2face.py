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

from models import create_model
import yaml
from os.path import join
import copy
from torch.utils.data import DataLoader
import config as cfg
import argparse
from models.feature2face_model import Feature2FaceModel
from options.train_feature2face_options import TrainOptions as RenderOptions
from options.train_audio2headpose_options import TrainOptions as HeadposeOptions
from options.train_audio2feature_options import TrainOptions as FeatureOptions
from tqdm import tqdm
from datasets.face_dataset import FaceDataset
import torch
import traceback
from funcs import utils
from util.visualizer import Visualizer


def train():
    f_option = FeatureOptions()
    h_option = HeadposeOptions()
    r_option = RenderOptions()

    # --continue_train --load_epoch 0 --epoch_count 0
    # --load_pretrain xxx --debug --fp16 1 --local_rank 1 --verbose
    #--continue_train --TTUR --no_html
    # seq_max_len not use
    args_raw = '--task Feature2Face --model feature2face --name Feature2Face --tf_log --gpu_ids 0\
        --dataset_mode face --dataset_names Vic --dataroot ./data \
        --isH5 1 --serial_batches --resize_or_crop scaleWidth  --no_flip 1 --suffix vic\
       --display_freq 100 --print_freq 10 --save_latest_freq 10 --save_epoch_freq 1 \
        --phase train --load_epoch latest --n_epochs_warm_up 5 \
        --n_epochs 5 --n_epochs_decay 5 --lr_decay_iters 1000 --lr_decay_gamma 0.25\
        --beta1 0.9 --lr 1e-4 --lr_final 1e-5 --lr_policy linear --gan_mode ls --pool_size 1 --frame_jump 1 \
        --epoch_count 0'
    args = utils.parse_args_str(args_raw)
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
    print("dataset image num:", dataset.__len__)

    with open(join('./config_file/', opt.dataset_names[0] + '.yaml')) as f:
        config = yaml.full_load(f)
    data_root = opt.dataroot
    opt.size = config['model_params']['Image2Image']['size']

    print('---------- Create Model: {} -------------'.format(opt.task))
    #model = Feature2FaceModel()
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    for i_epoch in range(opt.n_epochs):
        iter_cnt = 0
        train_iter = iter(dataloader)
        print(f'i_epoch: {i_epoch}')
        while 1:
            try:
                batch = next(train_iter)
                model.set_input(data=batch)
                model.optimize_parameters()
                runned_iter = iter_cnt + 1
                if runned_iter % opt.display_freq == 0 or iter_cnt == 0:
                    visualizer.display_current_results()
                if iter_cnt % opt.print_freq == 0:
                    print(f'iter_cnt: {iter_cnt}, loss_dict: {model.loss_dict}')
                iter_cnt += 1
            except Exception as e:
                trace_info = traceback.format_exc()
                print("exception: {}, trace_info: {}".format(e, trace_info))
                if len(model.loss_dict) == 0:
                    print("exception: {}, trace_info: {}".format(e, trace_info))
                model.loss_dict = {}
                break
        runned_epoch = i_epoch + 1
        if runned_epoch % opt.save_epoch_freq == 0 or i_epoch == 0:

            val_iter = iter(val_dataloader)
            while 1:
                try:
                    val_batch = next(val_iter)
                    model.set_input(data=batch)
                    model.validate()
                    print(f'val loss_dict: {model.loss_dict}')
                except Exception as e:
                    trace_info = traceback.format_exc()
                    if len(model.loss_dict) == 0:
                        print("exception: {}, trace_info: {}".format(e, trace_info))
                    model.loss_dict = {}
                    break
            # save mode
            model.save_networks(runned_epoch)


def main():
    train()


if __name__ == '__main__':
    main()
