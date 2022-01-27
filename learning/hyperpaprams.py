import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from genericpath import exists
from tqdm import tqdm
import scipy.io as sio
from funcs import utils
import h5py
import cv2
import io
import os
from os.path import join
from pathlib import Path
import torch
from skimage.io import imread, imsave
from PIL import Image
import bisect
import librosa
import platform
sys_name = platform.system()
from models.networks import APC_encoder
import config as cfg
import numpy as np
np.set_printoptions(suppress=1)
if sys_name != "Windows":
    camera_dir = "/data1/share/revolution_model/ExpressionTransmission/LiveSpeechPortraits/data/Vic"
else:
    camera_dir = r"E:\Topic\ExpressionTransmission\LiveSpeechPortraits\data\Vic"


def load_change_paras(person_dir):
    change_paras_name = 'change_paras.npz'
    change_paras_path = join(person_dir, change_paras_name)
    print(change_paras_path)
    change_paras = np.load(change_paras_path)
    scale, xc, yc = change_paras['scale'], change_paras['xc'], change_paras['yc']
    if cfg.origin_image_size[1] > cfg.origin_image_size[0]:
        scale = cfg.target_image_size[0] / cfg.origin_image_size[0]
        xc = round(cfg.origin_image_size[0] * scale * 0.5)
        yc = round(cfg.origin_image_size[1] * scale * 0.5)
    else:
        assert 0
    #scale, xc, yc = 1.0, 256, 256
    np.savez(change_paras_path, scale=scale, xc=xc, yc=yc)


def load_tracked_normalized_pts(person_dir):
    mean_pts3d_name = 'mean_pts3d.npy'
    mean_pts3d_path = join(person_dir, mean_pts3d_name)
    print(mean_pts3d_path)

    mean_pts3d = np.load(mean_pts3d_path)
    #mean_pts3d = pts_3d.mean(0)
    np.save(mean_pts3d_path, mean_pts3d)

    tracked3D_normalized_pts_fix_contour_name = 'tracked3D_normalized_pts_fix_contour.npy'
    tracked3D_normalized_pts_fix_contour_path = join(person_dir, tracked3D_normalized_pts_fix_contour_name)
    tracked3D_normalized_pts_fix_contour = np.load(tracked3D_normalized_pts_fix_contour_path)
    #tracked3D_normalized_pts_fix_contour = pts_3d
    np.save(tracked3D_normalized_pts_fix_contour_path, tracked3D_normalized_pts_fix_contour)

    racked2D_normalized_pts_fix_contour_name = 'tracked2D_normalized_pts_fix_contour.npy'
    racked2D_normalized_pts_fix_contour_path = join(person_dir, racked2D_normalized_pts_fix_contour_name)
    racked2D_normalized_pts_fix_contour = np.load(racked2D_normalized_pts_fix_contour_path)
    np.save(racked2D_normalized_pts_fix_contour_path, racked2D_normalized_pts_fix_contour)


def normalize(data, idxes, im_size, size, size_z, zero_center, per_channel):
    #idxes = [0,1,2]
    if per_channel:
        for idx in idxes:
            #data[:, :, idx] = (data[:, :, idx] - data[:, :, idx].min()) / (data[:, :, idx].max() - data[:, :, idx].min())
            data[:, :, idx] = data[:, :, idx] / im_size[idx]

            if zero_center:
                data[:, :, idx] = 2.0 * (data[:, :, idx] - 0.5)
                #data[:, :, idx] = 0.75 * data[:, :, idx]
            if idx != 2:
                data[:, :, idx] = size * data[:, :, idx]
            else:
                data[:, :, idx] = size_z * data[:, :, idx]
    else:
        data = (data - data.min()) / (data.max() - data.min())
        if zero_center:
            data = 2.0 * (data - 0.5)
            data = 0.75 * data
        data = size * data
    return data


def load_3d_fit_data_and_normalize(person_dir):
    fit_data_3d_name = '3d_fit_data.npz'
    fit_data_3d_path = join(person_dir, fit_data_3d_name)
    print(fit_data_3d_path)
    fit_data_3d = np.load(fit_data_3d_path)

    #pts_3d = fit_data_3d['pts_3d']

    #idxes = [0, 1, 2]
    #pts_3d = normalize(pts_3d, idxes, im_size=(1080, 1440), size=1.0, size_z=0.55, zero_center=1, per_channel=1)
    #pts_3d[:, :, 0] = -pts_3d[:, :, 0]
    #pts_3d[:, :, 1] = -pts_3d[:, :, 1]

    # for idx in idxes:
    #    pts_3d[:, :, idx] = (pts_3d[:, :, idx] - pts_3d[:, :, idx].min()) / (pts_3d[:, :, idx].max() - pts_3d[:, :, idx].min())
    #    pts_3d[:, :, idx] = 2.0 * (pts_3d[:, :, idx] - 0.5)
    #    pts_3d[:, :, idx] = 0.75 * pts_3d[:, :, idx]

    #mean_pts3d_name = 'mean_pts3d.npy'
    #mean_pts3d_path = join(person_dir, mean_pts3d_name)
    #mean_pts3d = np.load(mean_pts3d_path)
    #mean_pts3d = pts_3d.mean(0)
    #np.save(mean_pts3d_path, mean_pts3d)

    #tracked3D_normalized_pts_fix_contour_name = 'tracked3D_normalized_pts_fix_contour.npy'
    #tracked3D_normalized_pts_fix_contour_path = join(person_dir, tracked3D_normalized_pts_fix_contour_name)
    #tracked3D_normalized_pts_fix_contour = np.load(tracked3D_normalized_pts_fix_contour_path)
    #tracked3D_normalized_pts_fix_contour = pts_3d
    #np.save(tracked3D_normalized_pts_fix_contour_path, tracked3D_normalized_pts_fix_contour)

    #racked2D_normalized_pts_fix_contour_name = 'tracked2D_normalized_pts_fix_contour.npy'
    #racked2D_normalized_pts_fix_contour_path = join(person_dir, racked2D_normalized_pts_fix_contour_name)
    #racked2D_normalized_pts_fix_contour = np.load(racked2D_normalized_pts_fix_contour_path)
    #idxes = [0,1]
    #racked2D_normalized_pts_fix_contour = normalize(racked2D_normalized_pts_fix_contour,idxes, cfg.target_image_size[0], zero_center=0, per_channel=0)
    #np.save(racked2D_normalized_pts_fix_contour_path, racked2D_normalized_pts_fix_contour)

    #rot_angles = fit_data_3d['rot_angles']
    #mean_rot_angle = np.array([185.87375, -2.4958076, 1.2802227], dtype=np.float64)
    #for i_rot_angle in range(rot_angles.shape[0]):
    #    rot_angles[i_rot_angle] = mean_rot_angle
    #trans = fit_data_3d['trans']
    #mean_tran = np.array([-4.499857, 11.183616, 913.1682], dtype=np.float64)
    #for i_tran in range(trans.shape[0]):
    #    trans[i_tran] = mean_tran[:, None]

    #np.savez(fit_data_3d_path, pts_3d=pts_3d, rot_angles=rot_angles, trans=trans)

    show_3d_fit_data(person_dir, fit_data_3d)


def show_3d_fit_data(person_dir, fit_data_3d):
    camera, camera_intrinsic, scale = load_camera_info(camera_dir)

    pts_3d = fit_data_3d['pts_3d']
    rot_angles = fit_data_3d['rot_angles']
    trans = fit_data_3d['trans']
    nframe = len(fit_data_3d['pts_3d'])

    file_names = os.listdir(person_dir)

    interactive_mode = 1
    for i_nframe in tqdm(range(nframe)):
        im_path = join(person_dir, f'{i_nframe}.jpg')
        if os.path.exists(im_path):
            im = cv2.imread(im_path)
            im = cv2.resize(im, cfg.target_image_size)
            #utils.show_image(im, points=None, wait=1, name='im', channel_reverse=0)

        pt_3d = pts_3d[i_nframe]
        rot_angle = rot_angles[i_nframe]
        rot = utils.angle2matrix(rot_angle)
        tran = trans[i_nframe]
        #pts3d_headpose = scale * rot.dot(pt_3d.T) + tran
        #pts3d_viewpoint = camera.relative_rotation.dot(pts3d_headpose) + camera.relative_translation[:, None]
        #pts2d_project = camera_intrinsic.dot(pts3d_viewpoint)
        #pts2d_project[:2, :] /= pts2d_project[2, :]  # divide z
        #pts2d_project = pts2d_project[:2, :].T

        pts2d_project = pt_3d[:, :2]
        pts2d_project = (pts2d_project + 1.0) / 2.0
        pts2d_project[:, 0] *= cfg.target_image_size[0]
        pts2d_project[:, 1] *= cfg.target_image_size[1]
        pts2d_project = pts2d_project.astype(np.int)

        rot_show = utils.angle2matrix(np.array([90, 0, 0]))
        pt_3d_show = rot_show.dot(pt_3d.T).T
        #pt_3d_show = pt_3d
        #if utils.first_time:
        #    utils.show_pointcloud(pt_3d_show, use_pytorch3d=0, use_plt_loop=1, block=None, use_interactive_mode=1)
        #else:
        #    utils.verts_loop = pt_3d_show

        if os.path.exists(im_path):
            utils.show_image(im, points=pts2d_project, wait=0, name='im', channel_reverse=0)
        else:
            #canvas = np.zeros((1024, 1024, 3))
            canvas = np.zeros((cfg.target_image_size[1], cfg.target_image_size[0], 3))
            utils.show_image(canvas, points=pts2d_project, wait=1)
            del canvas
        pass


def load_clip_h5_file(person_dir):
    dataset_root = person_dir
    state = 'train'
    if state == 'train':
        clip_names = ['clip_0', 'clip_1', 'clip_2', 'clip_3']
    elif state == 'val':
        clip_names = ['clip_0']
    elif state == 'test':
        clip_names = ['clip_0']
    print(dataset_root)

    for clip_name in clip_names:
        im_dir = join(dataset_root, clip_name)
        file_names = os.listdir(im_dir)
        im_names = []
        for file_name in file_names:
            if file_name.endswith('.jpg'):
                im_names.append(file_name)
        im_names = sorted(im_names, key=lambda x: int(x.split('.jpg')[0]))
        im_paths = []
        for im_name in im_names:
            if im_name.endswith('.jpg'):
                im_path = join(im_dir, im_name)
                im_paths.append(im_path)

        clip_root = os.path.join(dataset_root, clip_name)
        img_file_path = os.path.join(clip_root, clip_name + '.h5')
        if os.path.exists(img_file_path):
            os.remove(img_file_path)
        f = h5py.File(img_file_path, "w")
        print(f'{clip_name} im_paths num: {len(im_paths)}')
        dset = f.create_dataset(clip_name, data=im_paths)

        img_file = h5py.File(img_file_path, 'r')[clip_name]
        continue
        for im_path in img_file:

            #byteImgIO = io.BytesIO()
            #byteImg = Image.open(im_path)
            #byteImg.save(byteImgIO, "PNG")
            # byteImgIO.seek(0)
            #byteImg = byteImgIO.read()
            #dataBytesIO = io.BytesIO(byteImg)
            # Image.open(dataBytesIO)

            #example = np.asarray(Image.open(io.BytesIO(im_path)))
            example = np.asarray(Image.open(im_path))
            h, w, _ = example.shape
            pass


def load_camera_info(camera_dir):
    camera = utils.camera()
    camera_intrinsic = np.load(join(camera_dir, 'camera_intrinsic.npy')).astype(np.float32)
    scale = sio.loadmat(join(camera_dir, 'id_scale.mat'))['scale'][0, 0]
    return camera, camera_intrinsic, scale


def load_APC_feature(person_dir):
    APC_feat_database = np.load(join(person_dir, 'APC_feature_base.npy'))

    name = 'train'
    APC_name = 'APC_epoch_160.model'
    APC_feature_file = name + '_APC_feature_{}.npy'.format(APC_name)
    pass


if __name__ == '__main__':
    if sys_name != "Windows":
        person_dir = camera_dir
    else:
        person_dir = camera_dir
        person_dir = join(camera_dir, 'clip_0')
    #load_change_paras(person_dir)
    #load_clip_h5_file(person_dir)
    load_3d_fit_data_and_normalize(person_dir)
    # load_tracked_normalized_pts(person_dir)
    pass
