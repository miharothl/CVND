import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}
    
class RandomHorizontalFlip(object):
    """Horizontally flip the given image randomly with a given probability.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, sample):
        image, key_pts_org = sample['image'], sample['keypoints']
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            
            image = np.copy(image)

            image = np.fliplr(image)
            
            key_pts_copy = np.copy(key_pts_org)
                        
            key_pts_copy[:,0] = -key_pts_copy[:,0]
            
            key_pts_copy[:,0] = key_pts_copy[:, 0] + image.shape[1]
            
            key_pts = np.copy(key_pts_copy)
            
            # mirror jawline 
            key_pts_copy[16] = key_pts[0]
            key_pts_copy[15] = key_pts[1]
            key_pts_copy[14] = key_pts[2]
            key_pts_copy[13] = key_pts[3]
            key_pts_copy[12] = key_pts[4]
            key_pts_copy[11] = key_pts[5]
            key_pts_copy[10] = key_pts[6]
            key_pts_copy[9] = key_pts[7]
            key_pts_copy[8] = key_pts[8]
            key_pts_copy[7] = key_pts[9]
            key_pts_copy[6] = key_pts[10]
            key_pts_copy[5] = key_pts[11]
            key_pts_copy[4] = key_pts[12]
            key_pts_copy[3] = key_pts[13]
            key_pts_copy[2] = key_pts[14]
            key_pts_copy[1] = key_pts[15]
            key_pts_copy[0] = key_pts[16]

            # mirror eyebrowns
            key_pts_copy[26] = key_pts[17]
            key_pts_copy[25] = key_pts[18]
            key_pts_copy[24] = key_pts[19]
            key_pts_copy[23] = key_pts[20]
            key_pts_copy[22] = key_pts[21]
            key_pts_copy[21] = key_pts[22]
            key_pts_copy[20] = key_pts[23]
            key_pts_copy[19] = key_pts[24]
            key_pts_copy[18] = key_pts[25]
            key_pts_copy[17] = key_pts[26]

            # mirror nose tip
            key_pts_copy[35] = key_pts[31]
            key_pts_copy[34] = key_pts[32]
            key_pts_copy[33] = key_pts[33]
            key_pts_copy[32] = key_pts[34]
            key_pts_copy[31] = key_pts[35]

            # mirror eyes
            key_pts_copy[45] = key_pts[36]
            key_pts_copy[44] = key_pts[37]
            key_pts_copy[43] = key_pts[38]
            key_pts_copy[42] = key_pts[39]
            key_pts_copy[47] = key_pts[40]
            key_pts_copy[46] = key_pts[41]
            key_pts_copy[39] = key_pts[42]
            key_pts_copy[38] = key_pts[43]
            key_pts_copy[37] = key_pts[44]
            key_pts_copy[36] = key_pts[45]
            key_pts_copy[41] = key_pts[46]
            key_pts_copy[40] = key_pts[47]

            # mirror lips
            key_pts_copy[54] = key_pts[48]
            key_pts_copy[53] = key_pts[49]
            key_pts_copy[52] = key_pts[50]
            key_pts_copy[51] = key_pts[51]
            key_pts_copy[50] = key_pts[52]
            key_pts_copy[49] = key_pts[53]
            key_pts_copy[48] = key_pts[54]

            key_pts_copy[59] = key_pts[55]
            key_pts_copy[58] = key_pts[56]
            key_pts_copy[57] = key_pts[57]
            key_pts_copy[56] = key_pts[58]
            key_pts_copy[55] = key_pts[59]

            key_pts_copy[64] = key_pts[60]
            key_pts_copy[63] = key_pts[61]
            key_pts_copy[62] = key_pts[62]
            key_pts_copy[61] = key_pts[63]
            key_pts_copy[60] = key_pts[64]

            key_pts_copy[67] = key_pts[65]
            key_pts_copy[66] = key_pts[66]
            key_pts_copy[65] = key_pts[67]

            return {'image': image, 'keypoints': key_pts_copy}

        return {'image': image, 'keypoints': key_pts_org}
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    
                
class RandomCropFace(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, padding):
        self.padding = padding

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
                
        tl_x = key_pts[:,0].min()
        tl_y = key_pts[:,1].min()

        br_x = key_pts[:,0].max()
        br_y = key_pts[:,1].max()
        
        h, w = image.shape[:2]

        new_h = br_y - tl_y
        new_w = br_x - tl_x
                
        h_pad_t = np.random.randint(0, self.padding)
        w_pad_l = np.random.randint(0, self.padding)
        
        if tl_y > h_pad_t:
            top = int(tl_y-h_pad_t)
        else:
            top = int(tl_y)
        
        if tl_x > w_pad_l:
            left = int(tl_x-w_pad_l)
        else:
            left = int(tl_x)
           
        if top < 0:
            top = 0
            
        if left < 0:
            left = 0
        
        cropped_height = int(top + new_h + self.padding)
        if cropped_height > h:
            cropped_height = int(top + new_h)
            
        if cropped_height > h:
            cropped_height = h
            
        cropped_width = int(left + new_w + self.padding)
        if cropped_width > w:
            cropped_width = int(left + new_w)
            
        if cropped_width > w:
            cropped_width = w
                
        image = image[top: cropped_height,
                      left: cropped_width]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}