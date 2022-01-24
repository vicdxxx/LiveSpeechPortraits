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
    to = HeadposeOptions()
    # --continue_train --load_epoch 0 --epoch_count 0
    args_raw = ''
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
                model.save_networks(i_epoch)

def main():
    train()


if __name__ == '__main__':
    main()
