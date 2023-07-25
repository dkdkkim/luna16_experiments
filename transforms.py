import numpy as np
from scipy.ndimage import rotate
import random

def normalize_uint8(inp: np.ndarray):
    output = inp / 255.
    return output

def unsqueeze(inp: np.ndarray):
    output = np.expand_dims(inp, axis=0)
    return output

def center_crop(inp: np.ndarray,
                crop_size: list):
    shp = inp.shape
    output = inp[int(shp[0]/2 - crop_size[0]/2):int(shp[0]/2 + crop_size[0]/2),
             int(shp[1]/2 - crop_size[1]/2):int(shp[1]/2 + crop_size[1]/2),
             int(shp[2]/2 - crop_size[2]/2):int(shp[2]/2 + crop_size[2]/2)]
    return output

def shift_crop(inp: np.ndarray,
                crop_size: list,
                shift: list)-> np.ndarray:
    shp = inp.shape
    output = inp[int(shp[0]/2 - crop_size[0]/2 + shift[0]):int(shp[0]/2 + crop_size[0]/2 + shift[0]),
             int(shp[1]/2 - crop_size[1]/2 + shift[1]):int(shp[1]/2 + crop_size[1]/2 + shift[1]),
             int(shp[2]/2 - crop_size[2]/2 + shift[2]):int(shp[2]/2 + crop_size[2]/2 + shift[2])]
    return output

def shift_crop_2d(inp: np.ndarray,
                  crop_size: list,
                  shift: list) -> np.ndarray:
    shp = inp.shape
    return inp[int(shp[0]/2 - crop_size[0]/2 + shift[0]):int(shp[0]/2 + crop_size[0]/2 + shift[0]),
             int(shp[1]/2 - crop_size[1]/2 + shift[1]):int(shp[1]/2 + crop_size[1]/2 + shift[1])]

def center_crop_2d(inp: np.ndarray,
                   crop_size: list,
                   multislice: bool):
    shp = inp.shape
    if multislice:
        return inp[:, int(shp[1]/2 - crop_size[0]/2):int(shp[1]/2 + crop_size[0]/2),
                int(shp[2]/2 - crop_size[1]/2):int(shp[2]/2 + crop_size[1]/2)]
    else:
        return inp[int(shp[0]/2 - crop_size[0]/2):int(shp[0]/2 + crop_size[0]/2),
                int(shp[1]/2 - crop_size[1]/2):int(shp[1]/2 + crop_size[1]/2)]

def rotate_yaxis(inp: np.ndarray,
                angle: int) -> np.ndarray:
    return rotate(inp, angle, axes=(0, 2), reshape=False)

def rotate_yaxis_position(inp: np.ndarray, position: np.ndarray,
                angle: int) -> tuple:
    if angle == 0:
        return rotate(inp, angle, axes=(0, 2), reshape=False), position
    elif angle == 180:
        position[0] = 1. - position[0]
        position[2] = 1. - position[2]
        return rotate(inp, angle, axes=(0, 2), reshape=False), position


def flip_yplane(inp: np.ndarray,
                flip_mode: int):
    if flip_mode == 0:
        return inp
    elif flip_mode == 1:
        return np.flip(inp, 0)
    elif flip_mode == 2:
        return np.flip(inp, 2)
    elif flip_mode == 3:
        return np.flip(np.flip(inp, 0), 2)

def flip_horizontal_2d(inp: np.ndarray):
    return inp[:,::-1]
    

def flip_yplane_position(inp: np.ndarray, position: np.ndarray,
                flip_mode: int):
    if flip_mode == 0:
        return inp, position
    elif flip_mode == 1:
        position[0] = 1. - position[0]
        return np.flip(inp, 0), position
    elif flip_mode == 2:
        position[2] = 1. - position[2]
        return np.flip(inp, 2), position
    elif flip_mode == 3:
        position[0] = 1. - position[0]
        position[2] = 1. - position[2]
        return np.flip(np.flip(inp, 0), 2), position

class Flip:
    """ Flip image on coronal plane """
    def __init__(self,
                transform_input=True,
                transform_target=False,
                is_mask=False,
                 is_position=False):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.is_mask=is_mask
        self.is_position=is_position
    
    def __call__(self, image, target, mask=None, position=None):
        flip_mode = random.sample([0, 1, 2, 3], 1)[0]
        flip_img = flip_yplane(image, flip_mode)
        if self.is_mask:
            if self.is_position:
                flip_mask, flip_shift = flip_yplane_position(mask, position)
                return flip_img, target, flip_mask, flip_shift
            else:
                flip_mask = flip_yplane(mask, flip_mode)
                return flip_img, target, flip_mask
        else:
            return flip_img, target
    
    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class Flip2D:
    """ Flip image on 2d image"""
    def __init__(self):
        pass
    
    def __call__(self, image, target):
        is_flip = random.sample([True, False], 1)[0]
        if is_flip:
            return flip_horizontal_2d(image), target
        else:
            return image, target

class FlipReg:
    """ Flip image on coronal plane """

    def __init__(self,
                 is_mask=False,
                 is_position=False):
        self.is_mask = is_mask
        self.is_position = is_position

    def __call__(self, image, target, mask=None, position=None):
        flip_mode = random.sample([0, 1, 2, 3], 1)[0]
        flip_img = flip_yplane(image, flip_mode)
        if flip_mode == 1:
            target[0] = -target[0]
        elif flip_mode == 2:
            target[2] = -target[2]
        elif flip_mode == 3:
            target[0] = -target[0]
            target[2] = -target[2]
        return flip_img, target

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class Rotate:
    """ Rotate images on y axis """

    def __init__(self,
                transform_input=True,
                transform_target=False,
                is_mask=False,
                 is_position=False,
                angles=[0, 180]):
                # angles=[0, 90, 180, 270]):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.is_mask=is_mask
        self.is_position = is_position
        self.angles = angles
    
    def __call__(self, inp, target, mask=None, position=None):
        angle = random.sample(self.angles, 1)[0]
        rot_img = rotate_yaxis(inp, angle)
        if self.is_mask:
            if self.is_position:
                rot_mask, rot_position = rotate_yaxis_position(mask,position, angle)
                return rot_img, target, rot_mask, rot_position
            else:
                rot_mask = rotate_yaxis(mask, angle)
                return rot_img, target, rot_mask
        else:
            return rot_img, target
    
    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class RotateReg:
    """ Rotate images on y axis """
    def __init__(self,
                 angles=[0, 180]):
        self.angles = angles

    def __call__(self, inp, target, mask=None, position=None):
        angle = random.sample(self.angles, 1)[0]
        rot_img = rotate_yaxis(inp, angle)
        if angle == 180:
            target = [-target[0], target[1], -target[2]]
        return rot_img, target

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class Normalize:
    """Normalize uint type images"""

    def __init__(self,
              transform_input=True,
              transform_target=False,
              is_mask=False,
                is_position=False):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.is_mask = is_mask
        self.is_position = is_position

    def __call__(self, inp, target, mask=None, position=None):
        output = normalize_uint8(inp)
        if self.is_mask:
            if self.is_position:
                return output, target, mask, position
            else:
                return output, target, mask
        else:
            return output, target

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class Unsqueeze:
    """Unsqueeze uint type images"""

    def __init__(self,
              transform_input=True,
              transform_target=False,
              is_mask=False,
                 is_position=False):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.is_mask = is_mask
        self.is_position=is_position

    def __call__(self, inp, target, mask=None, position=None):
        inp = unsqueeze(inp)
        if self.is_mask:
            # mask = unsqueeze(mask)
            if self.is_position:
                return inp, target, mask, position
            else:
                return inp, target, mask
        else:
            return inp,target

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class UnsqueezeReg:
    """Unsqueeze uint type images"""

    def __init__(self,
              transform_input=True,
              transform_target=False):
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __call__(self, inp, target, mask=None, position=None):
        inp = unsqueeze(inp)
        target = unsqueeze(target)
        return inp,target

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class CenterCrop:
    """Crop image on center position"""

    def __init__(self,
              transform_input=True,
              transform_target=False,
              crop_size:list=[48,24,48],
              is_mask=False):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.cs = crop_size
        self.cs_mask = [int(i/2) for i in crop_size]
        self.is_mask = is_mask

    def __call__(self, image, target, mask=None):
        image = center_crop(image,self.cs)
        if self.is_mask:
            mask = center_crop(mask, self.cs_mask)
            return image, target, mask
        else:
            return image, target

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class RandomCrop:
    """Crop image on ramdom position near center"""

    def __init__(self,
              transform_input=True,
              transform_target=False,
              crop_size:list=[48,24,48],
              is_mask=False,
              shift_ratio=0.05):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.cs = crop_size
        self.cs_mask = [int(i/2) for i in crop_size]
        self.is_mask = is_mask
        self.shift_ratio = shift_ratio

    def __call__(self, image, target, mask=None):
        shift = [random.sample(list(range(-int(self.cs[0]*self.shift_ratio),int(self.cs[0]*self.shift_ratio)+1)),1)[0],
                random.sample(list(range(-int(self.cs[1]*self.shift_ratio),int(self.cs[1]*self.shift_ratio)+1)),1)[0],
                random.sample(list(range(-int(self.cs[2]*self.shift_ratio),int(self.cs[2]*self.shift_ratio)+1)),1)[0]]
        image = shift_crop(image,self.cs, shift)
        if self.is_mask:
            mask_shift = [int(i/2) for i in shift]
            mask = shift_crop(mask, self.cs_mask, mask_shift)
            return image, target, mask
        else:
            return image, target

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class RandomCropReg:
    """Crop image on ramdom position near center"""

    def __init__(self,
              crop_size:list=[48,48,48],
             shift_ratio=8):
        self.cs = crop_size
        self.shift_ratio = shift_ratio

    def __call__(self, image):
        shift = [random.sample(list(range(-int(self.cs[0]/self.shift_ratio),int(self.cs[0]/self.shift_ratio)+1)),1)[0],
                random.sample(list(range(-int(self.cs[1]/self.shift_ratio),int(self.cs[1]/self.shift_ratio)+1)),1)[0],
                random.sample(list(range(-int(self.cs[2]/self.shift_ratio),int(self.cs[2]/self.shift_ratio)+1)),1)[0]]
        image = shift_crop(image,self.cs, shift)
        target = np.array([shift[0]/self.cs[0], shift[1]/self.cs[1], shift[2]/self.cs[2]])
        return image, target

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class RandomCropPosition:
    """Crop image on ramdom position near center"""

    def __init__(self,
              transform_input=True,
              transform_target=False,
              crop_size:list=[48,24,48],
              is_mask=False):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.cs = crop_size
        self.cs_mask = [int(i/2) for i in crop_size]
        self.is_mask = is_mask

    def __call__(self, image, target, mask=None):
        shift = [random.sample(list(range(-int(self.cs[0]/8),int(self.cs[0]/8)+1)),1)[0],
                random.sample(list(range(-int(self.cs[1]/8),int(self.cs[1]/8)+1)),1)[0],
                random.sample(list(range(-int(self.cs[2]/8),int(self.cs[2]/8)+1)),1)[0]]
        shift_label = (np.array(shift) + np.array(self.cs)*0.5) / np.array(self.cs)
        image = shift_crop(image,self.cs, shift)
        if self.is_mask:
            mask_shift = [int(i/2) for i in shift]
            mask = shift_crop(mask, self.cs_mask, mask_shift)
            return image, target, mask, shift_label
        else:
            return image, target, shift_label

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class CenterCropSlice:
    """ Select center slice and crop on center position"""
    def __init__(self,
                 plane_type: str,
                 crop_size: list
                 ) -> None:
        self.cs = crop_size
        if plane_type == 'transverse':
            self.plane_idx = 0
        elif plane_type == 'coronal':
            self.plane_idx = 1
        elif plane_type == 'sagittal':
            self.plane_idx = 2
        else:
            print('Wrong plane type')
            raise ValueError
    
    def __call__(self, image, target):
        shp = image.shape
        if self.plane_idx == 0:
            slice = image[shp[self.plane_idx]//2]
        elif self.plane_idx == 1:
            slice = image[:,shp[self.plane_idx]//2,:]
        elif self.plane_idx == 2:
            slice = image[...,shp[self.plane_idx]//2]
        img = center_crop_2d(slice, self.cs)
        return img, target
    
class CenterCrop2D:
    """ Crop on center position with 2D image"""
    def __init__(self,
                 crop_size: list
                 ) -> None:
        self.cs = crop_size
    
    def __call__(self, image, target):
        return center_crop_2d(image, self.cs, False), target
    
class RandomCropSlice:
    """ Select slice and crop on random position near center """
    
    def __init__(self,
                 crop_size:list,
                 plane_type:str,
                 n_slices:int = 3,
                 shift_ratio:float = 0.1
                 ) -> None:
        self.cs = crop_size
        self.shift_ratio = shift_ratio
        self.n_slices = n_slices
        if plane_type == 'transverse':
            self.plane_idx = 0
        elif plane_type == 'coronal':
            self.plane_idx = 1
        elif plane_type == 'sagittal':
            self.plane_idx = 2
        else:
            print('Wrong plane type')
            raise ValueError
    
    def __call__(self, image, target):
        shp = image.shape
        slice_idx = random.sample(list(range(shp[self.plane_idx]//2 - self.n_slices, shp[self.plane_idx]//2 + self.n_slices)), 1)[0]
        if self.plane_idx == 0:
            slice = image[slice_idx]
        elif self.plane_idx == 1:
            slice = image[:,slice_idx,:]
        elif self.plane_idx == 2:
            slice = image[...,slice_idx]
        shift = [random.sample(list(range(-int(self.cs[0] * self.shift_ratio),int(self.cs[0] * self.shift_ratio)+1)),1)[0],
                random.sample(list(range(-int(self.cs[1] * self.shift_ratio),int(self.cs[1] * self.shift_ratio)+1)),1)[0]]
        img = shift_crop_2d(slice, self.cs, shift)
        return  img, target

class RandomCrop2D:
    """ Random position near center with 2D image"""
    
    def __init__(self,
                 crop_size:list,
                 shift_ratio:float = 0.1
                 ) -> None:
        self.cs = crop_size
        self.shift_ratio = shift_ratio
    
    def __call__(self, image, target):
        shift = [random.sample(list(range(-int(self.cs[0] * self.shift_ratio),int(self.cs[0] * self.shift_ratio)+1)),1)[0],
                random.sample(list(range(-int(self.cs[1] * self.shift_ratio),int(self.cs[1] * self.shift_ratio)+1)),1)[0]]
        img = shift_crop_2d(image, self.cs, shift)
        return  img, target

class Compose:
    """Compose several transforms together"""
    def __init__(self,
                 transforms: list,
                 is_mask=False,
                 is_position=False
                 ):
        self.transforms = transforms
        self.is_mask = is_mask
        self.is_position = is_position
    def __call__(self, inp, target, mask=None, position=None):
        if self.is_mask:
            if self.is_position:
                for t in self.transforms:
                    if position == None:
                        inp, target, mask, position = t(inp, target, mask)
                    else:
                        inp, target, mask, position = t(inp, target, mask, position)
                return inp, target, mask, position
            for t in self.transforms:
                inp, target, mask = t(inp, target, mask)
            return inp, target, mask
        else:
            for t in self.transforms:
                inp, target = t(inp, target)
            return inp, target
    def __repr__(self): return str([transform for transform in self.transforms])

class ComposeReg:
    """Compose several transforms together"""
    def __init__(self,
                 transforms: list,
                 is_mask=False,
                 is_position=False
                 ):
        self.transforms = transforms
        self.is_mask = is_mask
        self.is_position = is_position

    def __call__(self, inp, mask=None, position=None):
        for idx, t in enumerate(self.transforms):
            if idx == 0:
                inp, target = t(inp)
            else:
                inp, target = t(inp, target)
        return inp, target

    def __repr__(self): return str([transform for transform in self.transforms])

class CenterCropMultiview:
    """ Select center slice and crop on center position"""
    def __init__(self,
                 crop_size: list,
                 multislice: bool
                 ) -> None:
        self.cs = crop_size
        self.multislice = multislice
    
    def __call__(self, image, target):
        shp = image.shape
        if self.multislice:
            trans_img = image[shp[0]//2-1:shp[0]//2+2]
            coron_img = image[:,shp[1]//2-1:shp[1]//2+2,:]
            coron_img = np.swapaxes(coron_img,0,1)
            sagit_img = image[...,shp[2]//2-1:shp[2]//2+2]
            sagit_img = np.transpose(sagit_img, (2,0,1))
        else:
            trans_img = image[shp[0]//2]
            coron_img = image[:,shp[1]//2,:]
            sagit_img = image[...,shp[2]//2]
        # print(trans_img.shape, coron_img.shape, sagit_img.shape)
        trans_img = center_crop_2d(trans_img, self.cs[1:], multislice=self.multislice)
        coron_img = center_crop_2d(coron_img, [self.cs[0], self.cs[2]], multislice=self.multislice)
        sagit_img = center_crop_2d(sagit_img, self.cs[:2], multislice=self.multislice)
        # print(trans_img.shape, coron_img.shape, sagit_img.shape)
        return [trans_img, coron_img, sagit_img], target

class RandomCropMultiview:
    """ Select center slice and crop on center position"""
    def __init__(self,
                 crop_size: list,
                 multislice: bool,
                 x_range: int,
                 y_range: int,
                 z_range: int
                 ) -> None:
        self.cs = crop_size
        self.multislice = multislice
        self.x_range = list(range((-1)*x_range,x_range))
        self.y_range = list(range((-1)*y_range,y_range))
        self.z_range = list(range((-1)*z_range,z_range))
    
    def __call__(self, image, target):
        shp = image.shape
        x_shift = random.sample(self.x_range,1)[0]
        y_shift = random.sample(self.y_range,1)[0]
        z_shift = random.sample(self.z_range,1)[0]
        if self.multislice:
            trans_img = image[shp[0]//2+z_shift-1:shp[0]//2+z_shift+2]
            coron_img = image[:,shp[1]//2+y_shift-1:shp[1]//2+y_shift+2,:]
            coron_img = np.swapaxes(coron_img,0,1)
            sagit_img = image[...,shp[2]//2+x_shift-1:shp[2]//2+x_shift+2]
            sagit_img = np.transpose(sagit_img, (2,0,1))
        else:
            trans_img = image[shp[0]//2+z_shift]
            coron_img = image[:,shp[1]//2+y_shift,:]
            sagit_img = image[...,shp[2]//2+x_shift]
        # print(trans_img.shape, coron_img.shape, sagit_img.shape)
        trans_img = center_crop_2d(trans_img, self.cs[1:], multislice=self.multislice)
        coron_img = center_crop_2d(coron_img, [self.cs[0], self.cs[2]], multislice=self.multislice)
        sagit_img = center_crop_2d(sagit_img, self.cs[:2], multislice=self.multislice)
        # print(trans_img.shape, coron_img.shape, sagit_img.shape)
        return [trans_img, coron_img, sagit_img], target
    
class NormalizeMultiview:
    """Normalize uint type images"""

    def __init__(self):
        pass

    def __call__(self, inp, target):
        return [normalize_uint8(x) for x in inp], target

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class UnsqueezeMultiview:
    """Unsqueeze uint type images"""

    def __init__(self):
        pass

    def __call__(self, inp, target, mask=None, position=None):
        return [unsqueeze(x) for x in inp], target

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})
