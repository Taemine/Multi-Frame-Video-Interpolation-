from PIL import ImageChops, Image
import math
import numpy as np
import os
import glob



def compute_IE_for_single_video(ground_truth_frame_paths, inference_frames_paths):
    """
    Assuming the ground_truth_frame and inference_frame have same shape,
    whose both height and width are divisible by 32
    
    Both two passing-in parameters are arrays of ordered jpeg images corresponding to each other for one video
    E.g. ["G_01.jpeg", "G_02.jpeg", "G_02.jpeg"] and ["I_01.jpeg", "I_02.jpeg", "I_02.jpeg"]

    Return: a IE(RMSE) score for this video --> float
    """

    img_g_t = []
    i = 0
    for filenames in sorted(glob.glob(os.path.join(ground_truth_frame_paths, '*.jpg'))):
        img_g_t.append(Image.open(filenames))
        i += 1

    img_infer = []
    i = 0
    for filenames in sorted(glob.glob(os.path.join(inference_frames_paths, '*.jpg'))):
        img_infer.append(Image.open(filenames))
        i += 1

    sum_rms = 0
    for i in range(len(img_infer)):
        sum_rms += rmsdiff(img_infer[i], img_g_t[i])
    IE = sum_rms / len(img_infer)
    return IE

def rmsdiff(im1, im2):
    """
    Calculates the root mean square error (RSME) between two images
    """
    errors = np.asarray(ImageChops.difference(im1, im2)) / 255
    return math.sqrt(np.mean(np.square(errors)))

