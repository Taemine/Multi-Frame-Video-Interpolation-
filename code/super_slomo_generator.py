import os

import torch
import torchvision.transforms as transforms
from PIL import Image

import model
import utility

if __name__ == '__main__':

    '''
    define our device as the first visible cuda device if we have CUDA available
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    '''
    initialize models
    '''
    flow_computation = model.UNet(6, 4)
    flow_computation.to(device)
    for param in flow_computation.parameters():
        param.requires_grad = False
    arbitrary_time_flow_interpolation = model.UNet(20, 5)
    arbitrary_time_flow_interpolation.to(device)
    for param in arbitrary_time_flow_interpolation.parameters():
        param.requires_grad = False

    '''
    load trained checkpoint
    '''
    checkpoint_path = '/Users/posoo/Box/Courses/SPRING-19/CSE676-Deep-Learning/Project/SuperSloMo.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    flow_computation.load_state_dict(checkpoint['state_dictFC'])
    arbitrary_time_flow_interpolation.load_state_dict(checkpoint['state_dictAT'])

    '''
    define file paths including the extracted frames path and ffmpeg path
    '''
    input_path = '/Users/posoo/Box/Courses/SPRING-19/CSE676-Deep-Learning/Project/test/cxk_25fps.mp4'
    output_path = '/Users/posoo/Box/Courses/SPRING-19/CSE676-Deep-Learning/Project/test/cxk_slomo.mp4'
    frames_dir_path = '/Users/posoo/Box/Courses/SPRING-19/CSE676-Deep-Learning/Project/test/frames'
    if not os.path.exists(frames_dir_path):
        os.mkdir(frames_dir_path)
    input_frames_dir_path = os.path.join(frames_dir_path, 'input')
    output_frames_dir_path = os.path.join(frames_dir_path, 'output')
    # os.mkdir(input_frames_dir_path)
    # os.mkdir(output_frames_dir_path)
    ffmpeg_path = '/Users/posoo/ffmpeg/ffmpeg'

    ffmpeg_utility = utility.FFmpegUtility(ffmpeg_path)
    # if ffmpeg_utility.extract_frames(input_path, input_frames_dir_path):
    #     logging.error("Failed to extract the frames of input video")

    transform_to_Tensor = transforms.Compose([transforms.ToTensor()])
    transform_to_PIL = transforms.Compose([transforms.ToPILImage()])

    reference_frames = utility.SingleSlomoDataset(input_frames_dir_path, transform_to_Tensor)
    reference_frames_loader = torch.utils.data.DataLoader(reference_frames, batch_size=1, shuffle=False)

    SPEEDUP = 3
    OUTPUT_VIDEO_FPS = 25
    # BATCH_SIZE = 2

    backward_warping = model.backWarp(reference_frames.frame_shape[0], reference_frames.frame_shape[1], device)

    with torch.no_grad():
        '''
        Start interpolation
        '''
        cnt = 0
        for _, (I_0, I_1) in enumerate(reference_frames_loader, 0):
            print(cnt)
            I_0 = I_0.to(device)
            I_1 = I_1.to(device)

            '''
            frames will cropped during the loading process.
            So rescale them to the original size while saving.
            '''
            I_0_saved = transform_to_PIL(I_0[0]).resize(reference_frames.original_shape, Image.BILINEAR)
            I_0_saved.save(os.path.join(output_frames_dir_path, '{:06d}.jpg'.format(cnt)))
            cnt += 1

            flow_outputs = flow_computation(torch.cat((I_0, I_1), dim=1))
            F_0_1 = flow_outputs[:, :2, :, :]
            F_1_0 = flow_outputs[:, 2:, :, :]

            """
            Init a SuperSlomo model which is like a computational graph adapting arbitrary_time_flow_interpolation_unet
            to interpolate intermediate frames
            """
            slomo_cp = model.SuperSlomo(arbitrary_time_flow_interpolation, backward_warping, I_0, I_1, device, F_0_1,
                                        F_1_0)
            interpolated = slomo_cp.forward(SPEEDUP)
            for I_t in interpolated:
                I_t_saved = transform_to_PIL(I_t[0]).resize(reference_frames.original_shape, Image.BILINEAR)
                I_t_saved.save(os.path.join(output_frames_dir_path, '{:06d}.jpg'.format(cnt)))
                cnt += 1

            I_1_saved = transform_to_PIL(I_1[0]).resize(reference_frames.original_shape, Image.BILINEAR)
            I_1_saved.save(os.path.join(output_frames_dir_path, '{:06d}.jpg'.format(cnt)))
            cnt += 1

    """
    Use FFmpeg to generate video from frames
    """
    ffmpeg_utility.create_video(output_frames_dir_path, output_path, OUTPUT_VIDEO_FPS)
