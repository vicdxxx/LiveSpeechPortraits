import os
from os.path import join
from pathlib import Path
import torch
from skimage.io import imread, imsave
from PIL import Image
import bisect
import numpy as np
np.set_printoptions(suppress=1)
import io
import cv2
import h5py
from funcs import utils
import scipy.io as sio
from tqdm import tqdm


def load_change_paras(person_dir):
    change_paras_name = 'change_paras.npz'
    change_paras_path = join(person_dir, change_paras_name)
    change_paras = np.load(change_paras_path)
    scale, xc, yc = change_paras['scale'], change_paras['xc'], change_paras['yc']
    scale, xc, yc = 1.0, 256, 256
    np.savez(change_paras_path, scale=scale, xc=xc, yc=yc)


def load_tracked_normalized_pts(person_dir):
    racked2D_normalized_pts_fix_contour_name = 'tracked2D_normalized_pts_fix_contour.npy'
    racked2D_normalized_pts_fix_contour_path = join(person_dir, racked2D_normalized_pts_fix_contour_name)
    racked2D_normalized_pts_fix_contour = np.load(racked2D_normalized_pts_fix_contour_path)
    np.save(racked2D_normalized_pts_fix_contour_path, racked2D_normalized_pts_fix_contour)

def load_3d_fit_data(person_dir):
    fit_data_3d_name = '3d_fit_data.npz'
    fit_data_3d_path = join(person_dir, fit_data_3d_name)
    fit_data_3d = np.load(fit_data_3d_path)
    pts_3d = fit_data_3d['pts_3d']

    #idxes = [0,1,2]
    #for idx in idxes:
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

    rot_angles = fit_data_3d['rot_angles']
    trans = fit_data_3d['trans']
    np.savez(fit_data_3d_path, pts_3d=pts_3d, rot_angles=rot_angles, trans=trans)
    show_3d_fit_data(person_dir, fit_data_3d)


def show_3d_fit_data(person_dir, fit_data_3d):
    camera, camera_intrinsic, scale = load_camera_info(person_dir)
    pts_3d = fit_data_3d['pts_3d']
    rot_angles = fit_data_3d['rot_angles']
    trans = fit_data_3d['trans']
    nframe = len(fit_data_3d['pts_3d'])

    interactive_mode = 1
    for i_nframe in tqdm(range(nframe)):
        pt_3d = pts_3d[i_nframe]
        rot_angle = rot_angles[i_nframe]
        rot = utils.angle2matrix(rot_angle)
        tran = trans[i_nframe]
        pts3d_headpose = scale * rot.dot(pt_3d.T) + tran
        pts3d_viewpoint = camera.relative_rotation.dot(pts3d_headpose) + camera.relative_translation[:, None]
        pts2d_project = camera_intrinsic.dot(pts3d_viewpoint)
        pts2d_project[:2, :] /= pts2d_project[2, :]  # divide z
        pts2d_project = pts2d_project[:2, :].T

        rot_show = utils.angle2matrix(np.array([90, 0, 0]))
        pt_3d_show = rot_show.dot(pt_3d.T).T
        if utils.first_time:
            utils.show_pointcloud(pt_3d_show, use_pytorch3d=0, use_plt_loop=1, block=None, use_interactive_mode=1)
        else:
            utils.verts_loop = pt_3d_show

        canvas = np.zeros((512, 512, 3))
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

    im_dir = join(dataset_root, r"src\frames")
    im_names = os.listdir(im_dir)
    im_paths = []
    for im_name in im_names:
        im_path = join(im_dir, im_name)
        im_paths.append(im_path)
    clip_nums = len(clip_names)
    for clip_name in clip_names:
        clip_root = os.path.join(dataset_root, state)
        img_file_path = os.path.join(clip_root, clip_name + '.h5')
        f = h5py.File(img_file_path, "w")
        dset = f.create_dataset(clip_name, data=im_paths)

        img_file = h5py.File(img_file_path, 'r')[clip_name]
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


def load_camera_info(person_dir):
    camera = utils.camera()
    camera_intrinsic = np.load(join(person_dir, 'camera_intrinsic.npy')).astype(np.float32)
    scale = sio.loadmat(join(person_dir, 'id_scale.mat'))['scale'][0, 0]
    return camera, camera_intrinsic, scale


def load_APC_feature(person_dir):
    APC_feat_database = np.load(join(person_dir, 'APC_feature_base.npy'))

    name = 'train'
    APC_name = 'APC_epoch_160.model'
    APC_feature_file = name + '_APC_feature_{}.npy'.format(APC_name)
    pass


if __name__ == '__main__':
    person_dir = r"E:\Topic\ExpressionTransmission\LiveSpeechPortraits\data\Vic\clip_3"
    #load_clip_h5_file(person_dir)
    load_3d_fit_data(person_dir)
    #load_tracked_normalized_pts(person_dir)
    pass
