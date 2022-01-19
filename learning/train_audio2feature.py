import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.audio2feature_model import Audio2FeatureModel
import argparse
from options.test_audio2feature_options import TestOptions as FeatureOptions
from options.test_audio2headpose_options import TestOptions as HeadposeOptions
from options.test_feature2face_options import TestOptions as RenderOptions
from tqdm import tqdm
from datasets.face_dataset import FaceDataset
from datasets.audiovisual_dataset import AudioVisualDataset
import torch
from learning.util import save_model


def train():
    parser = argparse.ArgumentParser()
    to = FeatureOptions()
    # --continue_train --load_epoch 0 --epoch_count 0
    args_raw = '--task Audio2Feature --model audio2feature --dataset_mode audiovisual --name Audio2Feature --gpu_ids 0 \
        --dataset_names common_voice --dataroot ./data \
        --frame_jump_stride 4 --num_threads 0 --batch_size 32 --serial_batches \
        --audio_encoder APC --feature_decoder LSTM --loss L2 --sequence_length 240 --FPS 60 --sample_rate 16000 \
        --audioRF_history 60 --audioRF_future 0 --feature_dtype pts3d --ispts_norm 1 --use_delta_pts 1 --frame_future 18 \
        --predict_length 1 --only_mouth 1 --verbose --suffix vic \
        --save_epoch_freq 5 --save_by_iter --phase train --re_transform 0 \
        --train_dataset_names train_list.txt --validate_dataset_names val_list.txt \
        --n_epochs 200 --lr_policy linear --lr 1e-4 --lr_final 1e-5 --n_epochs_decay 200 \
        --validate_epoch 10 --loss_smooth_weight 0 --optimizer Adam'
    args_raw = args_raw.split(' ')
    args = []
    for x in args_raw:
        if len(x.strip()) > 0:
            args.append(x)
    opt = to.parse(args=args)

    epoch_num = 1
    iter_per_epoch = 10
    dataset = AudioVisualDataset()
    model = Audio2FeatureModel()
    adam_betas = (0.9, 0.999)
    lr = 1e-4
    model.schedulers

    opt.save_epoch_freq
    opt.save_by_iter
    opt.loss_smooth_weight
    for i_epoch in range(opt.n_epochs):
        for A2Hsamples, target_info in dataset:
            #model.resume_training()
            model.set_input(A2Hsamples, target_info)
            model.forward()
            model.backward()
            if i_epoch % opt.save_epoch_freq == 0:
                model.validate()
                # save mode


def main():
    train()


if __name__ == '__main__':
    main()
