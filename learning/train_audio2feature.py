import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.audio2feature_model import Audio2FeatureModel
import argparse
from options.train_audio2feature_options import TrainOptions as FeatureOptions
from options.train_audio2headpose_options import TrainOptions as HeadposeOptions
from options.train_feature2face_options import TrainOptions as RenderOptions
from tqdm import tqdm
from datasets.audiovisual_dataset import AudioVisualDataset
import torch
from models import create_model
import yaml
from os.path import join
from models.networks import APC_encoder
import config as cfg
import numpy as np
import librosa
from funcs import utils
from torch.utils.data import DataLoader
import copy

only_generate_apc_features = 0


def generate_APC_feature(opt, config, dataset_root):
    # already has code in AudioVisualDataset -> need_deepfeats
    device = torch.device(0)
    print('---------- Loading Model: APC-------------')
    APC_model = APC_encoder(config['model_params']['APC']['mel_dim'],
                            config['model_params']['APC']['hidden_size'],
                            config['model_params']['APC']['num_layers'],
                            config['model_params']['APC']['residual'])
    APC_model_path = config['model_params']['APC']['ckp_path']
    APC_model.load_state_dict(torch.load(APC_model_path), strict=False)
    if opt.device == 'cuda':
        APC_model.cuda()
    APC_model.eval()

    clip_names = ['clip_0', 'clip_1', 'clip_2', 'clip_3']
    for clip_name in clip_names:
        clip_root = join(dataset_root, clip_name)
        driving_audio = join(clip_root, clip_name+".mp3")
        # create the results folder
        audio_name = os.path.split(driving_audio)[1][:-4]
        save_root = join(dataset_root, 'Vic', opt.id, audio_name)
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        print('Processing audio: {} ...'.format(audio_name))
        # read audio
        audio, _ = librosa.load(driving_audio, sr=cfg.sr)
        total_frames = np.int32(audio.shape[0] / cfg.sr * cfg.FPS)

        # 1. compute APC features
        print('1. Computing APC features...')
        mel80 = utils.compute_mel_one_sequence(audio, device=opt.device)
        mel_nframe = mel80.shape[0]
        with torch.no_grad():
            length = torch.Tensor([mel_nframe])
            mel80_torch = torch.from_numpy(mel80.astype(np.float32)).to(device).unsqueeze(0)
            hidden_reps = APC_model.forward(mel80_torch, length)[0]   # [mel_nframe, cfg.audio_feature_size]
            hidden_reps = hidden_reps.cpu().numpy()
        audio_feats = hidden_reps

        APC_name = os.path.split(APC_model_path)[-1]
        APC_feature_file = clip_name + '_APC_feature_{}.npy'.format(APC_name)
        APC_feature_path = os.path.join(clip_root, APC_feature_file)
        np.save(APC_feature_path, audio_feats)


def merge_APC_features_per_clip_to_one_file(opt, config, dataset_root):
    clip_names = ['clip_0', 'clip_1', 'clip_2', 'clip_3']
    APC_model_path = config['model_params']['APC']['ckp_path']
    all_audio_feats = None
    for i_clip, clip_name in enumerate(clip_names):
        clip_root = join(dataset_root, clip_name)
        APC_name = os.path.split(APC_model_path)[-1]
        APC_feature_file = clip_name + '_APC_feature_{}.npy'.format(APC_name)
        APC_feature_path = os.path.join(clip_root, APC_feature_file)
        clip_audio_feats = np.load(APC_feature_path)
        if i_clip == 0:
            all_audio_feats = clip_audio_feats
        else:
            all_audio_feats = np.concatenate([all_audio_feats, clip_audio_feats], 0)
    all_APC_feature_path = join(dataset_root, 'APC_feature_base.npy')
    if os.path.exists(all_APC_feature_path):
        os.remove(all_APC_feature_path)
    np.save(all_APC_feature_path, all_audio_feats)


def train():
    f_option = FeatureOptions()
    h_option = HeadposeOptions()
    r_option = RenderOptions()
    # --continue_train --load_epoch 0 --epoch_count 0
    #  --sequence_length 240 --time_frame_length 240 --A2L_receptive_field 255
    #  --FPS 22 --sample_rate 16000
    args_raw = '--task Audio2Feature --model audio2feature --dataset_mode audiovisual --name Audio2Feature --gpu_ids 0 \
        --dataset_names Vic --dataroot ./data \
        --frame_jump_stride 4 --num_threads 0 --batch_size 32 --serial_batches \
        --audio_encoder APC --feature_decoder LSTM --loss L2 \
        --dataset_type train \
        --audioRF_future 0 --feature_dtype pts3d --ispts_norm 1 --use_delta_pts 1 \
        --predict_length 1 --only_mouth 1 --verbose --suffix vic \
        --save_epoch_freq 50 --phase train --re_transform 0 \
        --train_dataset_names train_list.txt --validate_dataset_names val_list.txt \
        --n_epochs 200 --lr_policy linear --lr 1e-4 --lr_final 1e-5 --n_epochs_decay 200 \
        --validate_epoch 10  --optimizer Adam'
    args = utils.parse_args_str(args_raw)
    f_option.isTrain = True
    opt = f_option.parse(args=args)

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
        config = yaml.full_load(f)

    if only_generate_apc_features:
        dataset_root = os.path.join(opt.dataroot, opt.dataset_names)
        #generate_APC_feature(opt, config, dataset_root)
        merge_APC_features_per_clip_to_one_file(opt, config, dataset_root)
        return

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
                #for A2Hsamples, target_info in dataset:
                #model.resume_training()
                #model.set_input(data=[A2Hsamples[None, :], target_info[None, :]])
                model.set_input(data=batch)
                model.optimize_parameters()
                iter_cnt += 1
            except Exception as e:
                #print(f'exception: {e}')
                print(f'iter_cnt: {iter_cnt}, loss: {model.loss}')
                break
        runned_epoch = i_epoch + 1
        if runned_epoch % opt.save_epoch_freq == 0 or i_epoch == 0:
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
