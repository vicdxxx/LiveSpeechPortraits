import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.audio2feature_model import Audio2FeatureModel
import argparse
from options.train_audio2feature_options import TrainOptions
from tqdm import tqdm
from datasets.face_dataset import FaceDataset
from datasets.audiovisual_dataset import AudioVisualDataset
import torch


def train():
    parser = argparse.ArgumentParser()
    to = TrainOptions()
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
    av_dataset = AudioVisualDataset()
    a2f_model = Audio2FeatureModel()
    adam_betas = (0.9, 0.999)
    lr = 1e-4
    a2f_model.schedulers

    opt.save_epoch_freq
    opt.save_by_iter
    opt.loss_smooth_weight
    for i_epoch in range(opt.n_epochs):
        for A2Hsamples, target_info in av_dataset:
            #a2f_model.resume_training()
            a2f_model.set_input(A2Hsamples, target_info)
            a2f_model.forward()
            a2f_model.backward()
            if i_epoch % opt.save_epoch_freq == 0:
                a2f_model.validate()
                # save mode

def save_model():
        pass
        #md_chk_pt_name = f'{cnst.output_root}checkpoint/{str(args.run_id)}/{str(i + 1).zfill(6)}_{alpha}.model'

        #chk_pt_dict = {'generator_running': g_running.state_dict(),
        #            'generator': generator.state_dict(),
        #            'g_optimizer': g_optimizer.state_dict(),
        #            'discriminator_flm': discriminator_flm.state_dict(),
        #            'd_optimizer_flm': d_optimizer_flm.state_dict()}

        #torch.save(chk_pt_dict, md_chk_pt_name)

        #np.savez(md_chk_pt_name.replace('.model', '.npz'), step=step, used_sampless=used_sampless, alpha=alpha,
        #        resolution=resolution)


def main():
    train()


if __name__ == '__main__':
    main()
