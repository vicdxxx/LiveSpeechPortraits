import torch
from datasets.audiovisual_dataset import AudioVisualDataset
from datasets.face_dataset import FaceDataset
from tqdm import tqdm
from options.train_audio2headpose_options import TrainOptions

def train():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()

    epoch_num=1
    iter_per_epoch =10
    av_dataset = AudioVisualDataset()

def main():
    train()

if __name__ == '__main__':
    main()
