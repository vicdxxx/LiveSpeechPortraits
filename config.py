import platform
import numpy as np
sys_name = platform.system()
DEBUG = 1
if sys_name != "Windows":
    DEBUG = False

# official / Vic
DATASET_NAME = 'Vic'

target_image_size = (512, 512)
net_hidden_size = 512
audio_feature_size = 512

if DATASET_NAME == 'official':
    # official setting
    # use 73 face landmarks
    origin_image_size = (1920, 1080)
    sequence_length = 240
    time_frame_length = 240
    A2L_receptive_field = 255
    A2H_receptive_field = 255
    frame_future = 18

    n_fft = 512
    audio_extension = '.wav'
    face_landmark_num = 73
    shoulder_landmark_num = 18
    mouth_feature_num = 25
    sr, FPS = 16000, 60
    mouth_indices = np.concatenate([np.arange(4, 11), np.arange(46, 64)])
    mouth_range = range(46, 64)

    A2L_GMM_ndim = len(mouth_indices)*3

    eye_brow_indices = [27, 65, 28, 68, 29, 67, 30, 66, 31, 72, 32, 69, 33, 70, 34, 71]
    eye_brow_indices = np.array(eye_brow_indices, np.int32)
    part_list = [[list(range(0, 15))],                                # contour
                 [[15, 16, 17, 18, 18, 19, 20, 15]],                         # right eyebrow
                 [[21, 22, 23, 24, 24, 25, 26, 21]],                         # left eyebrow
                 [range(35, 44)],                                     # nose
                 [[27, 65, 28, 68, 29], [29, 67, 30, 66, 27]],                # right eye
                 [[33, 69, 32, 72, 31], [31, 71, 34, 70, 33]],                # left eye
                 [range(46, 53), [52, 53, 54, 55, 56, 57, 46]],             # mouth
                 [[46, 63, 62, 61, 52], [52, 60, 59, 58, 46]]                 # tongue
                 ]
    mouth_outer = [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 46]
    label_list = [1, 1, 2, 3, 3, 4, 5]  # labeling for different facial parts

    upper_outer_lip = list(range(47, 52))
    upper_inner_lip = [63, 62, 61]
    upper_mouth = [46, 47, 48, 49, 50, 51, 52, 61, 62, 63]
    lower_inner_lip = [58, 59, 60]
    lower_outer_lip = list(range(57, 52, -1))
    lower_mouth = [53, 54, 55, 56, 57, 58, 59, 60]


elif DATASET_NAME == 'Vic':
    # new setting
    # use 68 face landmarks
    origin_image_size = (1080, 1440)
    sequence_length = 90
    time_frame_length = 90
    A2L_receptive_field = 100
    A2H_receptive_field = 100
    frame_future = 7

    A2L_GMM_ndim = 20*3
    n_fft = 1024
    audio_extension = '.mp3'
    face_landmark_num = 68
    shoulder_landmark_num = 0
    mouth_feature_num = 20
    sr, FPS = 16000, 22
    mouth_indices = np.concatenate([np.arange(48, 60), np.arange(60, 68)])
    mouth_range = range(48, 68)
    A2L_GMM_ndim = len(mouth_indices)*3

    eye_brow_indices = np.concatenate([np.arange(17, 22), np.arange(22, 27)])
    part_list = [[list(range(0, 17))],                                # contour
                 [list(range(17, 22))],                         # right eyebrow
                 [list(range(22, 27))],                         # left eyebrow
                 [list(range(27, 36))],                                     # nose
                 [list(range(36, 42))],                # right eye
                 [list(range(42, 48))],                # left eye
                 [list(range(48, 60))],             # mouth
                 [list(range(60, 68))]                 # tongue
                 ]
    mouth_outer = list(range(48, 60))
    label_list = [1, 1, 2, 3, 3, 4, 5]  # labeling for different facial parts

    upper_outer_lip = list(range(49, 53))
    upper_inner_lip = list(range(61, 63))
    lower_outer_lip = list(range(55, 59))
    lower_inner_lip = list(range(65, 67))
    upper_mouth = upper_outer_lip + upper_inner_lip
    lower_mouth = lower_outer_lip + lower_inner_lip

else:
    assert 0
