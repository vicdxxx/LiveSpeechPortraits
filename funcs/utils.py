import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import config as cfg
from . import audio_funcs
import numpy as np
from math import cos, sin
import torch
from numpy.linalg import solve
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import KDTree
import time
from tqdm import tqdm


class camera(object):
    def __init__(self, fx=0, fy=0, cx=0, cy=0):
        self.name = 'default camera'
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.relative_rotation = np.diag([1, 1, 1]).astype(np.float32)
        self.relative_translation = np.zeros(3, dtype=np.float32)
#        self.intrinsic = np.array([[self.fx, 0, self.cx],
#                                   [0, self.fy, self.cy],
#                                   [0,       0,       1]])

    def intrinsic(self, trans_matrix=0):
        ''' compute the intrinsic matrix
        '''
        intrinsic = np.array([[self.fx, 0, self.cx],
                              [0, self.fy, self.cy],
                              [0, 0, 1]])

        return intrinsic

    def relative(self):
        ''' compute the relative transformation 4x4 matrix with respect to the
        first camera kinect. specially the kinect's relative transformation
        matrix is exact a identity matrix.
        '''
        relative = np.eye(4, dtype=np.float32)
        relative[:3, :3] = self.relative_rotation
        relative[:3, 3] = self.relative_translation

        return relative

    def transform_intrinsic(self, transform_matrix):
        ''' change the camera intrinsic matrix
        transformed_intrinsic = transform_matrix * intrinsic
        '''
        scale = transform_matrix[0, 0]
        self.fx *= scale
        self.fy *= scale
        self.cx = scale * self.cx + transform_matrix[0, 2]
        self.cy = scale * self.cy + transform_matrix[1, 2]


def compute_mel_one_sequence(audio, hop_length=int(16000/(cfg.FPS*2)), winlen=1/cfg.FPS, winstep=0.5/cfg.FPS, sr=cfg.sr, fps=cfg.FPS, device='cpu'):
    ''' compute mel for an audio sequence.
    '''
    device = torch.device(device)
    Audio2Mel_torch = audio_funcs.Audio2Mel(n_fft=cfg.n_fft, hop_length=int(cfg.sr/(cfg.FPS*2)), win_length=int(cfg.sr/cfg.FPS), sampling_rate=cfg.sr,
                                            n_mel_channels=80, mel_fmin=90, mel_fmax=7600.0).to(device)

    nframe = int(audio.shape[0] / cfg.sr * cfg.FPS)
    mel_nframe = 2 * nframe
    mel_frame_len = int(sr * winlen)
    mel_frame_step = sr * winstep

    mel80s = np.zeros([mel_nframe, 80])
    for i in range(mel_nframe):
        #    for i in tqdm(range(mel_nframe)):
        st = int(i * mel_frame_step)
        audio_clip = audio[st: st + mel_frame_len]
        if len(audio_clip) < mel_frame_len:
            audio_clip = np.concatenate([audio_clip, np.zeros([mel_frame_len - len(audio_clip)])])
        audio_clip_device = torch.from_numpy(audio_clip).unsqueeze(0).unsqueeze(0).to(device).float()
        mel80s[i] = Audio2Mel_torch(audio_clip_device).cpu().numpy()[0].T   # [1, 80]

    return mel80s


def KNN(feats, feat_database, K=10):
    ''' compute KNN for feat in feat base
    '''
    tree = KDTree(feat_database, leaf_size=100000)
    print('start computing KNN ...')
    st = time.time()
    dist, ind = tree.query(feats, k=K)
    et = time.time()
    print('Taken time: ', et-st)

    return dist, ind


def KNN_with_torch(feats, feat_database, K=10):
    feats = torch.from_numpy(feats)  # .cuda()
    feat_database = torch.from_numpy(feat_database)  # .cuda()
    # Training
    feat_base_norm = (feat_database ** 2).sum(-1)
    #    print('start computing KNN ...')
    #    st = time.time()
    feats_norm = (feats ** 2).sum(-1)
    # Rely on cuBLAS for better performance!
    diss = (feats_norm.view(-1, 1) + feat_base_norm.view(1, -1) - 2 * feats @ feat_database.t())
    ind = diss.topk(K, dim=1, largest=False).indices
    #    et = time.time()
    #    print('Taken time: ', et-st)

    return ind.cpu().numpy()


def solve_LLE_projection(feat, feat_base):
    '''find LLE projection weights given feat base and target feat
    Args:
        feat: [ndim, ] target feat
        feat_base: [K, ndim] K-nearest feat base
    =======================================
    We need to solve the following function
    ```
        min|| feat - \sum_0^k{w_i} * feat_base_i ||, s.t. \sum_0^k{w_i}=1
    ```
    equals to:
        ft = w1*f1 + w2*f2 + ... + wk*fk, s.t. w1+w2+...+wk=1
           = (1-w2-...-wk)*f1 + w2*f2 + ... + wk*fk
     ft-f1 = w2*(f2-f1) + w3*(f3-f1) + ... + wk*(fk-f1)
     ft-f1 = (f2-f1, f3-f1, ..., fk-f1) dot (w2, w3, ..., wk).T
        B  = A dot w_,  here, B: [ndim,]  A: [ndim, k-1], w_: [k-1,]
    Finally,
       ft' = (1-w2-..wk, w2, ..., wk) dot (f1, f2, ..., fk)
    =======================================
    Returns:
        w: [K,] linear weights, sums to 1
        ft': [ndim,] reconstructed feats
    '''
    K, ndim = feat_base.shape
    if K == 1:
        feat_fuse = feat_base[0]
        w = np.array([1])
    else:
        w = np.zeros(K)
        B = feat - feat_base[0]   # [ndim,]
        A = (feat_base[1:] - feat_base[0]).T   # [ndim, K-1]
        AT = A.T
        w[1:] = solve(AT.dot(A), AT.dot(B))
        w[0] = 1 - w[1:].sum()
        feat_fuse = w.dot(feat_base)

    return w, feat_fuse


def compute_LLE_projection_frame(feats, feat_database, ind):
    nframe = feats.shape[0]
    feat_fuse = np.zeros_like(feats)
    w = np.zeros([nframe, ind.shape[1]])
    current_K_feats = feat_database[ind]
    w, feat_fuse = solve_LLE_projection(feats, current_K_feats)

    return w, feat_fuse


def compute_LLE_projection_all_frame(feats, feat_database, ind, nframe):
    nframe = feats.shape[0]
    feat_fuse = np.zeros_like(feats)
    w = np.zeros([nframe, ind.shape[1]])
    for i in tqdm(range(nframe), desc='LLE projection'):
        current_K_feats = feat_database[ind[i]]
        w[i], feat_fuse[i] = solve_LLE_projection(feats[i], current_K_feats)

    return w, feat_fuse


def angle2matrix(angles, gradient='false'):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left.
        z: roll. positive for tilting head right.
        gradient(str): whether to compute gradient matrix: dR/d_x,y,z
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx = np.array([[1, 0, 0],
                   [0, cos(x), -sin(x)],
                   [0, sin(x), cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, sin(y)],
                   [0, 1, 0],
                   [-sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), -sin(z), 0],
                   [sin(z), cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    # R=Rx.dot(Ry.dot(Rz))

    if gradient != 'true':
        return R.astype(np.float32)
    elif gradient == 'true':
        # gradident matrix
        dRxdx = np.array([[0, 0, 0],
                          [0, -sin(x), -cos(x)],
                          [0, cos(x), -sin(x)]])
        dRdx = Rz.dot(Ry.dot(dRxdx)) * np.pi/180
        dRydy = np.array([[-sin(y), 0, cos(y)],
                          [0, 0, 0],
                          [-cos(y), 0, -sin(y)]])
        dRdy = Rz.dot(dRydy.dot(Rx)) * np.pi/180
        dRzdz = np.array([[-sin(z), -cos(z), 0],
                          [cos(z), -sin(z), 0],
                          [0, 0, 0]])
        dRdz = dRzdz.dot(Ry.dot(Rx)) * np.pi/180

        return R.astype(np.float32), [dRdx.astype(np.float32), dRdy.astype(np.float32), dRdz.astype(np.float32)]


def project_landmarks(camera_intrinsic, viewpoint_R, viewpoint_T, scale, headposes, pts_3d):
    ''' project 2d landmarks given predicted 3d landmarks & headposes and user-defined
    camera & viewpoint parameters
    '''
    rot, trans = angle2matrix(headposes[:3]), headposes[3:][:, None]
    pts3d_headpose = scale * rot.dot(pts_3d.T) + trans
    pts3d_viewpoint = viewpoint_R.dot(pts3d_headpose) + viewpoint_T[:, None]
    pts2d_project = camera_intrinsic.dot(pts3d_viewpoint)
    pts2d_project[:2, :] /= pts2d_project[2, :]  # divide z
    pts2d_project = pts2d_project[:2, :].T

    return pts2d_project, rot, trans


def project_landmarks_orthogonal(camera_intrinsic, viewpoint_R, viewpoint_T, scale, headposes, pts_3d):
    ''' project 2d landmarks given predicted 3d landmarks & headposes and user-defined
    camera & viewpoint parameters
    '''
    pts2d_project = pts_3d[:, :, :2]
    pts2d_project = (pts2d_project + 1.0) / 2.0
    pts2d_project[:, :, 0] *= cfg.target_image_size[0]
    pts2d_project[:, :, 1] *= cfg.target_image_size[1]
    return pts2d_project, None, None


def landmark_smooth_3d(pts3d, smooth_sigma=0, area='only_mouth'):
    ''' smooth the input 3d landmarks using gaussian filters on each dimension.
    Args:
        pts3d: [N, cfg.face_landmark_num, 3]
    '''
    # per-landmark smooth
    if not smooth_sigma == 0:
        if area == 'all':
            pts3d = gaussian_filter1d(pts3d.reshape(-1, cfg.face_landmark_num*3), smooth_sigma, axis=0).reshape(-1, cfg.face_landmark_num, 3)
        elif area == 'only_mouth':
            mouth_pts3d = pts3d[:, cfg.mouth_range, :].copy()
            mouth_pts3d = gaussian_filter1d(mouth_pts3d.reshape(-1, 18*3), smooth_sigma, axis=0).reshape(-1, 18, 3)
            pts3d = gaussian_filter1d(pts3d.reshape(-1, cfg.face_landmark_num*3), smooth_sigma, axis=0).reshape(-1, cfg.face_landmark_num, 3)
            pts3d[:, cfg.mouth_range, :] = mouth_pts3d

    return pts3d


def mouth_pts_AMP(pts3d, is_delta=True, method='XY', paras=[1, 1]):
    ''' mouth region AMP to control the reaction amplitude.
    method: 'XY', 'delta', 'XYZ', 'LowerMore' or 'CloseSmall'
    '''
    if method == 'XY':
        AMP_scale_x, AMP_scale_y = paras
        if is_delta:
            pts3d[:, cfg.mouth_range, 0] *= AMP_scale_x
            pts3d[:, cfg.mouth_range, 1] *= AMP_scale_y
        else:
            mean_mouth3d_xy = pts3d[:, cfg.mouth_range, :2].mean(axis=0)
            pts3d[:, cfg.mouth_range, 0] += (AMP_scale_x-1) * (pts3d[:, cfg.mouth_range, 0] - mean_mouth3d_xy[:, 0])
            pts3d[:, cfg.mouth_range, 1] += (AMP_scale_y-1) * (pts3d[:, cfg.mouth_range, 1] - mean_mouth3d_xy[:, 1])
    elif method == 'delta':
        AMP_scale_x, AMP_scale_y = paras
        if is_delta:
            diff = AMP_scale_x * (pts3d[1:, cfg.mouth_range] - pts3d[:-1, cfg.mouth_range])
            pts3d[1:, cfg.mouth_range] += diff

    elif method == 'XYZ':
        AMP_scale_x, AMP_scale_y, AMP_scale_z = paras
        if is_delta:
            pts3d[:, cfg.mouth_range, 0] *= AMP_scale_x
            pts3d[:, cfg.mouth_range, 1] *= AMP_scale_y
            pts3d[:, cfg.mouth_range, 2] *= AMP_scale_z

    elif method == 'LowerMore':
        upper_x, upper_y, upper_z, lower_x, lower_y, lower_z = paras
        if is_delta:
            pts3d[:, cfg.upper_mouth, 0] *= upper_x
            pts3d[:, cfg.upper_mouth, 1] *= upper_y
            pts3d[:, cfg.upper_mouth, 2] *= upper_z
            pts3d[:, cfg.lower_mouth, 0] *= lower_x
            pts3d[:, cfg.lower_mouth, 1] *= lower_y
            pts3d[:, cfg.lower_mouth, 2] *= lower_z

    elif method == 'CloseSmall':
        open_x, open_y, open_z, close_x, close_y, close_z = paras
        nframe = pts3d.shape[0]
        for i in tqdm(range(nframe), desc='AMP mouth..'):
            if sum(pts3d[i, cfg.upper_mouth, 1] > 0) + sum(pts3d[i, cfg.lower_mouth, 1] < 0) > 16 * 0.3:
                # open
                pts3d[i, cfg.mouth_range, 0] *= open_x
                pts3d[i, cfg.mouth_range, 1] *= open_y
                pts3d[i, cfg.mouth_range, 2] *= open_z
            else:
                # close
                pts3d[:, cfg.mouth_range, 0] *= close_x
                pts3d[:, cfg.mouth_range, 1] *= close_y
                pts3d[:, cfg.mouth_range, 2] *= close_z

    return pts3d


def solve_intersect_mouth(pts3d):
    ''' solve the generated intersec lips, usually happens in mouth AMP usage.
    Args:
        pts3d: [N, cfg.face_landmark_num, 3]
    '''
    upper_inner = pts3d[:, cfg.upper_inner_lip]
    lower_inner = pts3d[:, cfg.lower_inner_lip]

    lower_inner_y = lower_inner[:, :, 1]
    upper_inner_y = upper_inner[:, :, 1]
    # all three inner lip flip
    flip = lower_inner_y > upper_inner_y
    flip = np.where(flip.sum(axis=1) == 3)[0]

    # flip frames
    inner_y_diff = lower_inner_y[flip] - upper_inner_y[flip]
    half_inner_y_diff = inner_y_diff * 0.5
    # upper inner
    pts3d[flip[:, None], cfg.upper_inner_lip, 1] += half_inner_y_diff
    # lower inner
    pts3d[flip[:, None], cfg.lower_inner_lip, 1] -= half_inner_y_diff
    # upper outer
    pts3d[flip[:, None], cfg.upper_outer_lip, 1] += half_inner_y_diff.mean()
    # lower outer
    pts3d[flip[:, None], cfg.lower_outer_lip, 1] -= half_inner_y_diff.mean()

    return pts3d


def headpose_smooth(headpose, smooth_sigmas=[0, 0], method='gaussian'):
    rot_sigma, trans_sigma = smooth_sigmas
    rot = gaussian_filter1d(headpose.reshape(-1, 6)[:, :3], rot_sigma, axis=0).reshape(-1, 3)
    trans = gaussian_filter1d(headpose.reshape(-1, 6)[:, 3:], trans_sigma, axis=0).reshape(-1, 3)
    headpose_smooth = np.concatenate([rot, trans], axis=1)

    return headpose_smooth


interactive_mode = 1
ax = None
verts_loop = None
first_time = True
def show_pointcloud_loop(arg=0):
    global interactive_mode
    global ax
    global verts_loop
    global first_time
    from matplotlib import pyplot as plt
    surf = ax.scatter(verts_loop[:, 0] * 1.2, verts_loop[:, 1], verts_loop[:, 2], c='cyan', alpha=1.0, edgecolor='b')
    if not interactive_mode:
        if first_time:
            plt.show()
            first_time = False
        else:
            plt.show(block=False)
    if interactive_mode:
        plt.pause(0.001)


def show_pointcloud(verts, use_pytorch3d=1, use_plt_loop=0, block=None, use_interactive_mode=1):
    if use_pytorch3d:
        import torch
        from pytorch3d.structures import Pointclouds
        from pytorch3d.vis.plotly_vis import AxisArgs, plot_scene
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        if not isinstance(verts, torch.Tensor):
            verts = torch.Tensor(verts).to(device)

        point_cloud = Pointclouds(points=[verts])
        point_cloud_person = {
            "3DMM": {
                "obj": point_cloud
            }
        }
        fig = plot_scene(
            point_cloud_person,
            xaxis={"backgroundcolor": "rgb(200, 200, 230)"},
            yaxis={"backgroundcolor": "rgb(230, 200, 200)"},
            zaxis={"backgroundcolor": "rgb(200, 230, 200)"},
            axis_args=AxisArgs(showgrid=True))
        fig.show()
    elif use_plt_loop:
        import pylab
        from matplotlib import pyplot as plt
        global interactive_mode
        global ax
        global verts_loop
        global first_time
        interactive_mode = use_interactive_mode
        if interactive_mode:
            plt.ion()
        import matplotlib
        matplotlib.use('TkAgg')
        fig = plt.figure(0, figsize=plt.figaspect(1.0))
        fig.canvas.mpl_connect('key_press_event', show_pointcloud_loop)
        verts_loop = verts
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(elev=0, azim=-90)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        show_pointcloud_loop()
    else:
        from matplotlib import pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')
        fig = plt.figure(figsize=plt.figaspect(.5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        surf = ax.scatter(verts[:, 0] * 1.2, verts[:, 1], verts[:, 2], c='cyan', alpha=1.0, edgecolor='b')
        plt.show()


def show_image(image, wait=0, name="x", channel_reverse=True, points=None):
    import numpy as np
    import cv2 as cv
    if isinstance(image, np.ndarray):
        img = image.copy()
    else:
        #import paddle
        #if isinstance(image, paddle.Tensor):
        #    img = image.numpy().copy()
        import torch
        if isinstance(image, torch.Tensor):
            img = image.cpu().detach().numpy().copy()

    if len(img.shape) == 4:
        if img.shape[1] == 3:
            img = img.transpose(0, 2, 3, 1)[0]
        if img.shape[1] == 1:
            img = image.transpose(0, 2, 3, 1)[0]
    elif len(img.shape) == 3:
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        elif img.shape[1] > img.shape[0] and img.shape[2] > img.shape[0]:
            img = img[0]
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    img = img.astype(np.uint8)
    img = np.squeeze(img)

    if points is not None:
        print("points num:", len(points))
        for point in points:
            point = (int(point[0]), int(point[1]))
            cv.circle(img, point, 3, (0, 255, 0), -1)
    if len(img.shape) == 3 and channel_reverse:
        img = img[:, :, ::-1]
    print(img.shape)
    cv.namedWindow(name, 0)
    cv.resizeWindow(name, cfg.target_image_size[0], cfg.target_image_size[1])
    cv.imshow(name, img)
    if wait is not None:
        cv.waitKey(wait)
