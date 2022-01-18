import torch
from datasets.audiovisual_dataset import AudioVisualDataset
from datasets.face_dataset import FaceDataset
from tqdm import tqdm
from options.train_feature2face_options import TrainOptions
from models.audio2headpose_model import Audio2FeatureModel
import argparse


def train():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()

    epoch_num = 1
    iter_per_epoch = 10
    face_dataset = FaceDataset()
    a2f_model = Audio2FeatureModel()
    adam_betas = (0.9, 0.999)
    lr = 1e-4
    a2f_model.schedulers

def main():
    train()


if __name__ == '__main__':
    main()
