import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.audio2feature_model import Audio2FeatureModel
import argparse
from options.test_audio2feature_options import TestOptions as FeatureOptions
from options.test_audio2headpose_options import TestOptions as HeadposeOptions
from options.test_feature2face_options import TestOptions as RenderOptions
from tqdm import tqdm
from datasets.audiovisual_dataset import AudioVisualDataset
import torch
from os.path import join
import numpy as np
import librosa
import config as cfg
from torch.utils.data import DataLoader
from funcs import utils
import copy
import yaml
from models import create_model


def default_headpose_from_McStay():
    person_dir = r"E:\Topic\ExpressionTransmission\LiveSpeechPortraits\data\Vic\clip_3"
    fit_data_3d_name = '3d_fit_data.npz'
    fit_data_3d_path = join(person_dir, fit_data_3d_name)
    fit_data_3d = np.load(fit_data_3d_path)
    rot_angles = fit_data_3d['rot_angles']
    mean_rot_angle = np.array([185.87375, -2.4958076, 1.2802227], dtype=np.float64)
    for i_rot_angle in range(rot_angles.shape[0]):
        rot_angles[i_rot_angle] = mean_rot_angle
    trans = fit_data_3d['trans']
    mean_tran = np.array([-4.499857, 11.183616, 913.1682], dtype=np.float64)
    for i_tran in range(trans.shape[0]):
        trans[i_tran] = mean_tran[:, None]


def train():
    f_option = FeatureOptions()
    h_option = HeadposeOptions()
    r_option = RenderOptions()
    # --save_by_iter --continue_train
    # --serial_batches --verbose
    # not use --re_transform --gaussian_noise 1 --gaussian_noise_scale 0.01
    args_raw = '--task Audio2Headpose --model audio2headpose --dataset_mode audiovisual \
        --name Audio2Headpose --gpu_ids 0 \
        --audioRF_future 0 --feature_decoder WaveNet --loss GMM \
        --dataset_names Vic --dataroot ./data --frame_jump_stride 1 --num_threads 0 --batch_size 32 \
        --audio_encoder APC --audiofeature_input_channels 80 \
        --predict_length 5 --audio_windows 2 -dataset_type train --suffix vic \
        --save_epoch_freq 50 --load_epoch 0 --epoch_count 0 --phase train \
        --smooth_loss 0 --n_epochs 500 --lr 1e-4 --lr_final 1e5 --n_epochs_decay 500 --validate_epoch 50 \
        --optimizer Adam'
    args = utils.parse_args_str(args_raw)
    h_option.isTrain = True
    opt = h_option.parse(args=args)

    opt.FPS = float(opt.FPS)
    assert opt.FPS == cfg.FPS

    dataset = AudioVisualDataset(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=True)
    val_opt = copy.deepcopy(opt)
    val_opt.dataset_type = 'val'
    val_dataset = AudioVisualDataset(val_opt)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=True)

    with open(join('./config_file/', opt.dataset_names + '.yaml')) as f:
        config = yaml.load(f)
    data_root = opt.dataroot

    print('---------- Create Model: {} -------------'.format(opt.task))
    #model = Audio2FeatureModel()
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
                #print(f'exception: {e}')
                print(f'iter_cnt: {iter_cnt}, loss: {model.loss}')
                break
        runned_epoch = i_epoch + 1
        if runned_epoch % opt.save_epoch_freq == 0:
            val_iter = iter(val_dataloader)
            while 1:
                try:
                    val_batch = next(val_iter)
                    model.set_input(data=batch)
                    model.validate()
                    print(f'val_batch loss: {model.loss}')
                except Exception as e:
                    #print(f'exception: {e}')
                    break
            # save mode
            model.save_networks(runned_epoch)

def main():
    train()


if __name__ == '__main__':
    main()
