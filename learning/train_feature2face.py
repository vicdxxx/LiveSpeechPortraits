"""
8 diﬀerent target sequences of 7 diﬀerent subjects for training and testing

All videos are extracted at 60 frames per second (FPS)
and the synchronized audio waves are sampled at 16KHz frequency

crop the video to keep the face at the center and then resize to 512 × 512

cfg.face_landmark_num pre-defned facial landmarks
"""
import torch
from datasets.face_dataset import FaceDataset
from tqdm import tqdm
from options.test_audio2feature_options import TestOptions as FeatureOptions
from options.test_audio2headpose_options import TestOptions as HeadposeOptions
from options.test_feature2face_options import TestOptions as RenderOptions
from models.feature2face_model import Feature2FaceModel
import argparse
from learning.util import save_model


def train():
    parser = argparse.ArgumentParser()
    to = RenderOptions()
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
    dataset = FaceDataset()
    model = Feature2FaceModel()
    model.schedulers

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
