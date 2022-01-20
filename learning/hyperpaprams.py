import os
from os.path import join
from pathlib import Path
import torch
from skimage.io import imread, imsave
from PIL import Image
import bisect
import numpy as np
import io
import cv2
import h5py

if __name__ == '__main__':
    person_dir = r"E:\Topic\ExpressionTransmission\LiveSpeechPortraits\data\Vic"
    change_paras_name = 'change_paras.npz'
    change_paras_path = join(person_dir, change_paras_name)
    change_paras = np.load(change_paras_path)
    scale, xc, yc = change_paras['scale'], change_paras['xc'], change_paras['yc']
    scale, xc, yc = 1.0, 256, 256
    np.savez(change_paras_path, scale=scale, xc=xc, yc=yc)

    mean_pts3d_name = 'mean_pts3d.npy'
    racked2D_normalized_pts_fix_contour_name = 'tracked2D_normalized_pts_fix_contour.npy'
    tracked3D_normalized_pts_fix_contour_name = 'tracked3D_normalized_pts_fix_contour.npy'
    mean_pts3d_path = join(person_dir, mean_pts3d_name)
    racked2D_normalized_pts_fix_contour_path = join(person_dir, racked2D_normalized_pts_fix_contour_name)
    tracked3D_normalized_pts_fix_contour_path = join(person_dir, tracked3D_normalized_pts_fix_contour_name)
    mean_pts3d = np.load(mean_pts3d_path)
    racked2D_normalized_pts_fix_contour = np.load(racked2D_normalized_pts_fix_contour_path)
    tracked3D_normalized_pts_fix_contour = np.load(tracked3D_normalized_pts_fix_contour_path)
    np.save(mean_pts3d_path, mean_pts3d)
    np.save(racked2D_normalized_pts_fix_contour_path, racked2D_normalized_pts_fix_contour)
    np.save(tracked3D_normalized_pts_fix_contour_path, tracked3D_normalized_pts_fix_contour)

    fit_data_3d_name = '3d_fit_data.npz'
    fit_data_3d_path = join(person_dir, fit_data_3d_name)
    fit_data_3d = np.load(fit_data_3d_path)
    pts_3d = fit_data_3d['pts_3d']
    rot_angles = fit_data_3d['rot_angles']
    trans = fit_data_3d['trans']
    np.savez(change_paras_path, pts_3d=pts_3d, rot_angles=rot_angles, trans=trans)
    pass
