import os
import random

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def open_image(image_path, resized_shape=None, crop_area=None, horizontal_ﬂip=False):
    """
    Opening image may accompany with some data augmentation
    :param image_path: image path
    :param resized_shape: resize image if specified
    :param crop_area: crop image if specified. It'a tuple of (left, up, right, bottom)
    :param horizontal_ﬂip: do horizontal flip if specified
    :return: a PIL.Image object
    """
    image = Image.open(image_path)
    image = image.resize(resized_shape, Image.ANTIALIAS) if (resized_shape is not None) else image
    image = image.transpose(Image.FLIP_LEFT_RIGHT) if horizontal_ﬂip else image
    image = image.crop(crop_area) if (crop_area is not None) else image

    return image


class FFmpegUtility(object):
    """
    This class serves to call FFmpeg
    """

    def __init__(self, ffmpeg_path) -> None:
        """
        :param ffmpeg_path: the path of executable ffmpeg file
        """
        super().__init__()
        self._ffmpeg_path = ffmpeg_path

    def extract_frames(self, video_input, extracted_frames_dir_path):
        cmd_str = '{} -i {} -vsync 0 -qscale:v 2 {}/%06d.jpg'.format(self._ffmpeg_path, video_input,
                                                                     extracted_frames_dir_path)
        return os.system(cmd_str)  # Return 0 if succeed

    def create_video(self, output_frames_dir_path, output_path, fps):
        cmd_str = '{} -r {} -i {}/%6d.jpg -crf 17 -vcodec libx264 {}'.format(self._ffmpeg_path, fps,
                                                                             output_frames_dir_path,
                                                                             output_path)
        return os.system(cmd_str)  # Return 0 if succeed


class SlomoDataset(Dataset):
    """
    This class serves for preparing the training/testing/validation data sets
    """

    def __init__(self, input_extracted_frames_dir_path, transform=None, need_augmentation=False):
        super().__init__()
        """
        self.frame_paths is a 2D array: [clip_index][frame_index]
        """
        self.transform = transform
        self.need_augmentation = need_augmentation
        self.frame_paths = []
        for clip_index, clip_dir in enumerate(os.listdir(input_extracted_frames_dir_path)):
            clip_dir_path = os.path.join(input_extracted_frames_dir_path, clip_dir)
            if os.path.isdir(clip_dir_path):
                self.frame_paths.append(
                    [os.path.join(clip_dir_path, frame) for frame in sorted(os.listdir(clip_dir_path))])

    def __getitem__(self, index):
        if self.need_augmentation:
            """augmentation part"""
        pass

    def __len__(self):
        return len(self.frame_paths)


class AdobeDataset(Dataset):
    """
    This class serves for preparing the Adobe 240fps Dataset to train and test model
    Each frame in the Adobe set has a shape of (640, 360).
    According to the paper, it will be randomly cropped to (352, 352) and horizontal flipped
    during the training process. However in the phase of validation or testing there is no need to do the cropping
    but should resize it to meet the requirements of U-Net.

    So, in design of this class, allow to pass-in these parameters in initialization.
    """

    def __init__(self, folder_path, crop_shape=None, train=True, transform=None):
        """
        :param folder_path: could be training/testing/validation folder
        """
        super().__init__()
        self.transform = transform
        self.train = train
        self.resized_shape = (360, 360)  # only be used in training
        self.crop_shape = crop_shape
        self.folder_path = folder_path

        """
        Create a 2D array to store the frames in each clip
        """
        clip_paths = [os.path.join(self.folder_path, str(clip))
                      for clip in sorted([int(c)
                                          for c in os.listdir(self.folder_path)
                                          if os.path.isdir(os.path.join(self.folder_path, c))])]

        self.frame_paths = []
        for clip_path in clip_paths:
            temp = sorted([os.path.join(clip_path, f) for f in os.listdir(clip_path) if f.endswith(".jpg")])
            if len(temp) > 2:
                self.frame_paths.append(temp)

    def __getitem__(self, index):
        """
        :param index:
        :return: a tuple of (I_0, I_t, I_1, t)
        """
        I_0_index = random.randint(0, 3)
        I_1_index = I_0_index + 8
        I_t_index_relative = random.randint(1, 7)
        I_t_index = I_t_index_relative + I_0_index
        t = I_t_index_relative / 8
        clip = self.frame_paths[index]
        temp = []
        if self.train:
            """
            For data augmentation, we randomly reverse the direction of entire sequence and select 9 consecutive frames 
            for training. On the image level, each video frame is re-sized to have a shorter spatial dimension of 360 
            and a random crop of 352 × 352 plus horizontal ﬂip are performed.
            """
            """
            randomly horizontal flip
            """
            flip = True if random.random() > 0.5 else False
            """
            randomly reversed order
            """
            if random.random() > 0.5:
                I_0_index, I_1_index = I_1_index, I_0_index
            left = random.randint(0, 8)
            up = random.randint(0, 8)
            right = left + 352
            bottom = up + 352
            crop_area = (left, up, right, bottom)
            for i in [I_0_index, I_t_index, I_1_index]:
                I_i = open_image(clip[i], resized_shape=self.resized_shape, crop_area=crop_area, horizontal_ﬂip=flip)
                I_i = self.transform(I_i) if self.transform is not None else I_i
                temp.append(I_i)
        else:
            """
            need implemented for testing/validation
            """
            for i in [I_0_index, I_t_index, I_1_index]:
                I_i = open_image(clip[i])
                I_i = self.transform(I_i) if self.transform is not None else I_i
                temp.append(I_i)
        return temp[0], temp[1], temp[2], t  # (I_0, I_t, I_1, t)

    def __len__(self):
        return len(self.frame_paths)

    def __add__(self, other):
        return super().__add__(other)


class SingleSlomoDataset(Dataset):
    """
    This class serves for preparing the data to generate slomo video
    """

    def __init__(self, extracted_frames_dir_path, transform=None):
        super().__init__()
        self.transform = transform  # must passing a transform to transform PIL to Tensor
        self.frame_paths = [os.path.join(extracted_frames_dir_path, frame) for frame in
                            sorted(os.listdir(extracted_frames_dir_path))]
        if len(self.frame_paths) == 0:
            raise (RuntimeError("Cannot find extracted video frames in  path {}".format(extracted_frames_dir_path)))

        """
        because we need to resize the frame to do the convolution,
        we need to record the original_shape and frame_shape
        """
        temp_frame = open_image(self.frame_paths[0])
        self.original_shape = temp_frame.size
        self.frame_shape = (self.original_shape[0] // 32 * 32, self.original_shape[1] // 32 * 32)

    def __getitem__(self, index):
        """

        :param index: the index of interpolation interval
        :return: [I_0, I_1] are two frames to inference and interpolate the intermediate frames
        """
        return [self.transform(open_image((self.frame_paths[i]), resized_shape=self.frame_shape)) for i in
                [index, index + 1]]

    def __len__(self):
        return len(self.frame_paths) - 1  # N_{intervals} = N_{frames} - 1


if __name__ == '__main__':
    # frames_dir_path = '/Users/posoo/GitRepos/adobeset/train'
    # slomoset = SlomoDataset(frames_dir_path)
    # oneVideoSet = SingleSlomoDataset("/Users/posoo/Box/Courses/SPRING-19/CSE676-Deep-Learning/Project/test/frames/input")
    # oneFramePair = oneVideoSet.__getitem__(0)
    transform = transforms.Compose([transforms.ToTensor()])
    adobeset = AdobeDataset("/Users/posoo/Box/Courses/SPRING-19/CSE676-Deep-Learning/Project/adobeset/train",
                            transform=transform)
    sample = adobeset.__getitem__(24)
    print(True)
