import random
import jittor as jt 
import cv2
import numpy as np
import math
from PIL import Image
from jdet.utils.registry import build_from_cfg,TRANSFORMS
from jdet.models.boxes.box_ops import rotated_box_to_poly_np,poly_to_rotated_box_np,norm_angle
from jdet.models.boxes.iou_calculator import bbox_overlaps_np
from numpy import random as nprandom

import mmcv
@TRANSFORMS.register_module()
class RRandomCrop:
    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 version='oc'):
        self.version = version
        super(RRandomCrop, self).__init__()
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }

    def filter_border(self, bboxes, h, w):
        """Filter the box whose center point is outside or whose side length is
        less than 5."""
       
        x_ctr, y_ctr = bboxes[:, 0], bboxes[:, 1]
        w_bbox, h_bbox = bboxes[:, 2], bboxes[:, 3]
        keep_inds = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h) & \
                    (w_bbox > 5) & (h_bbox > 5)
        return keep_inds

    def _crop_data(self, img, annos, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0

        margin_h = max(img.size[0] - crop_size[0], 0)
        margin_w = max(img.size[1] - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # crop the image
        img = np.array(img)
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape[:2]
        annos['img_size'] = img_shape

        height, width = img_shape

            # e.g. gt_bboxes and gt_bboxes_ignore
        bbox_offset = np.array([offset_w, offset_h, 0, 0, 0],
                                dtype=np.float32)
        bboxes = annos['rboxes'] - bbox_offset

        valid_inds = self.filter_border(bboxes,height, width)
        # If the crop does not contain any gt-bbox area and
        # allow_negative_crop is False, skip this image.
        # annos['rboxes'] = bboxes[valid_inds]
        polys = np.clip(rotated_box_to_poly_np(bboxes[valid_inds]), 0, 1023) 

        annos['rboxes'] = poly_to_rotated_box_np(polys, 'le90')
        annos['labels'] = annos['labels'][valid_inds]
        if annos.__contains__('polys'):
            poly_offset = np.array([offset_w, offset_h,offset_w, offset_h,offset_w, offset_h,offset_w, offset_h],
                                dtype=np.float32)
            annos['polys'] = polys
        if annos.__contains__('polys'):
            hbox_offset = np.array([offset_w, offset_h,0,0],
                                dtype=np.float32)
            annos['hboxes'] = annos['hboxes'][valid_inds] - hbox_offset
     

        return img, annos

    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.
        Args:
            image_size (tuple): (h, w).
        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == 'absolute_range':
            assert self.crop_size[0] <= self.crop_size[1]
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == 'relative_range':
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def __call__(self, img, annos):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        o_img = img.copy()
        o_annos = annos.copy()
        image_size = img.size
        crop_size = self._get_crop_size(image_size)
        image, anno = self._crop_data(img, annos, crop_size)
        if len(anno['labels']) == 0:
            return o_img, o_annos
        return Image.fromarray(image), anno

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        return repr_str

@TRANSFORMS.register_module()
class RandmGrayScale:
    def __init__(self, prob = 0.5) -> None:
        self.prob = prob
    
    def is_convert_gray(self):
        return np.random.rand() < self.prob

    def __call__(self, img, annos):
        if self.is_convert_gray():
            img = img.convert('L')
        
            img = np.array(img)
            img = Image.fromarray(np.stack((img,)*3, axis=-1))

        return img, annos

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gray_annos={self.annos}'
        return repr_str


@TRANSFORMS.register_module()
class Compose:
    def __init__(self, transforms=None):
        self.transforms = []
        if transforms is None:
            transforms = []
        for transform in transforms:
            if isinstance(transform,dict):
                transform = build_from_cfg(transform,TRANSFORMS)
            elif not callable(transform):
                raise TypeError('transform must be callable or a dict')
            self.transforms.append(transform)

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        
        return image, target

@TRANSFORMS.register_module()
class RandomRotateAug:
    def __init__(self, angle_version = 'le135', random_rotate_on=False):
        self.random_rotate_on = random_rotate_on
        self.angle_version = angle_version
    
    def _rotate_boxes_90(self,target,size):
        w,h = size
        for key in["bboxes","hboxes","rboxes","polys","hboxes_ignore","polys_ignore","rboxes_ignore"]:
            if key not in target:
                continue
            bboxes = target[key]
            if bboxes.ndim<2:
                continue
            if "bboxes" in key or "hboxes" in key:
                new_boxes = np.zeros_like(bboxes)
                new_boxes[:,  ::2] = bboxes[:, 1::2] # x = y
                # new_boxes[:, 1::2] = w - bboxes[:, -2::-2] # y = w - x
                new_boxes[:, 1] = w - bboxes[:, 2] # y = w - x
                new_boxes[:, 3] = w - bboxes[:, 0] # y = w - x
                target[key] = new_boxes
                continue

            if "rboxes" in key:
                bboxes  = rotated_box_to_poly_np(bboxes, self.angle_version)

            new_bboxes = np.zeros_like(bboxes)
            new_bboxes[:,0::2] = bboxes[:,1::2]
            new_bboxes[:,1::2] = w-bboxes[:,0::2]

            if "rboxes" in key:
                new_bboxes = poly_to_rotated_box_np(new_bboxes, self.angle_version)

            target[key]=new_bboxes

    def __call__( self, image, target=None ):
        # (0, 90, 180, or 270)
        if self.random_rotate_on:
            indx = int(random.random() * 100) // 25
            # anticlockwise
            for _ in range(indx):
                if target is not None:
                    self._rotate_boxes_90(target,image.size)
                image = image.rotate(90,expand=True)
            if target is not None:
                target["rotate_angle"]=90*indx

        return image, target


@TRANSFORMS.register_module()
class PolyRandomRotate(object):

    def __init__(self,
                 rotate_ratio=0.5,
                 mode='range',
                 angles_range=180,
                 auto_bound=False,
                 angle_version='le90'):
        self.rotate_ratio = rotate_ratio
        self.auto_bound = auto_bound
        assert mode in ['range', 'value'], \
            f"mode is supposed to be 'range' or 'value', but got {mode}."
        if mode == 'range':
            assert isinstance(angles_range, int), \
                "mode 'range' expects angle_range to be an int."
        else:
            assert mmcv.is_seq_of(angles_range, int) and len(angles_range), \
                "mode 'value' expects angle_range as a non-empty list of int."
        self.mode = mode
        self.angles_range = angles_range
        self.discrete_range = [90, 180, -90, -180]
        self.version = angle_version

    @property
    def is_rotate(self):
        """Randomly decide whether to rotate."""
        return np.random.rand() < self.rotate_ratio

    def apply_image(self, img, bound_h, bound_w, interp=cv2.INTER_LINEAR):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0:
            return img
        return cv2.warpAffine(
            img, self.rm_image, (bound_w, bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y)
        points
        """
        if len(coords) == 0:
            return coords
        coords = np.asarray(coords, dtype=float)
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def create_rotation_matrix(self,
                               center,
                               angle,
                               bound_h,
                               bound_w,
                               offset=0):
        """Create rotation matrix."""
        center += offset
        rm = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        if self.auto_bound:
            rot_im_center = cv2.transform(center[None, None, :] + offset,
                                          rm)[0, 0, :]
            new_center = np.array([bound_w / 2, bound_h / 2
                                   ]) + offset - rot_im_center
            rm[:, 2] += new_center
        return rm

    def filter_border(self, bboxes, h, w):
        """Filter the box whose center point is outside or whose side length is
        less than 5."""
       
        x_ctr, y_ctr = bboxes[:, 0], bboxes[:, 1]
        w_bbox, h_bbox = bboxes[:, 2], bboxes[:, 3]
        keep_inds = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h) & \
                    (w_bbox > 5) & (h_bbox > 5)
        return keep_inds

    def __call__(self, image, target=None ):
        """Call function of PolyRandomRotate."""
        o_img = image.copy()
        o_target = target.copy()
        h, w = image.size
        image = np.array(image, dtype=np.uint8)
        if not self.is_rotate:
            angle = 0
        else:
            if self.mode == 'range':
                angle = self.angles_range * (2 * np.random.rand() - 1)
            else:
                i = np.random.randint(len(self.angles_range))
                angle = self.angles_range[i]

        target['rotate_angle'] = angle

        image_center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = \
            abs(np.cos(angle / 180 * np.pi)), abs(np.sin(angle / 180 * np.pi))
        if self.auto_bound:
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos,
                 h * abs_cos + w * abs_sin]).astype(int)
        else:
            bound_w, bound_h = w, h

        self.rm_coords = self.create_rotation_matrix(image_center, angle,
                                                     bound_h, bound_w)
        self.rm_image = self.create_rotation_matrix(
            image_center, angle, bound_h, bound_w, offset=-0.5)

        image = self.apply_image(image, bound_h, bound_w)
        target['img_size'] = (bound_h, bound_w)
        gt_bboxes = target.get('rboxes', [])
        labels = target.get('labels', [])
        polys = rotated_box_to_poly_np(gt_bboxes).reshape(-1, 2)
        polys = self.apply_coords(polys).reshape(-1, 8)
        gt_bboxes = poly_to_rotated_box_np(polys,self.version) \
                
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        keep_inds = self.filter_border(gt_bboxes, bound_h, bound_w)
        gt_bboxes = gt_bboxes[keep_inds, :]
        polys = polys[keep_inds]
        max_xy = polys.reshape(-1,4,2).max(axis=1)
        min_xy = polys.reshape(-1,4,2).min(axis=1)
        # print(polys,max_xy, min_xy)
        res_hbox = []
        for each in zip(max_xy, min_xy):
            x,y = (each[0]+each[1])/2
            h,w = each[0]-each[1]
            res_hbox.append([x,y,h,w])
     
        hbox = np.array(res_hbox).astype(np.float32)
        
        labels = labels[keep_inds]
        
        target['rboxes'] = gt_bboxes
        target['labels'] = labels
        target['polys'] = polys
        target['hboxes'] = hbox
        # print("Image.fromarray(image) , target",Image.fromarray(image) , target)
        if len(labels) == 0:
            return o_img, o_target
        return Image.fromarray(image) , target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_ratio={self.rotate_ratio}, ' \
                    f'angles_range={self.angles_range}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@TRANSFORMS.register_module()
class Resize:
    def __init__(self, min_size, max_size, keep_ratio=True):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.keep_ratio = keep_ratio

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size

        if self.keep_ratio:
            # NOTE Mingtao
            if w <= h:
              size = np.clip( size, int(w / 1.5), int(w * 1.5) )
            else:
              size = np.clip( size, int(h / 1.5), int(h * 1.5) )

            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w),1.

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
            assert np.abs(oh/h - ow/w)<1e-2
            
        else:
            oh = self.min_size[0]
            ow = self.max_size
        
        return (oh, ow),oh/h

    def _resize_boxes(self,target,size):
        for key in ["bboxes","polys"]:
            if key not in target:
                continue
            bboxes = target[key]
            width,height = target["img_size"]
            new_w,new_h = size
            bboxes[:,0::2] = bboxes[:,0::2]*float(new_w/width)
            bboxes[:,1::2] = bboxes[:,1::2]*float(new_h/height)

            # clip to border
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, new_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, new_h - 1)

            target[key]=bboxes
    
    def _resize_mask(self,target,size):
        pass

    def __call__(self, image, target=None):
        size,scale_factor = self.get_size(image.size)
        image = image.resize(size[::-1],Image.BILINEAR)
        if target is not None:
            self._resize_boxes(target,image.size)
            target["img_size"] = image.size
            target["scale_factor"] = scale_factor
            target["pad_shape"] = image.size
            target["keep_ratio"] = self.keep_ratio
        return image, target

@TRANSFORMS.register_module()
class MinIoURandomCrop:
    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3,
                 bbox_clip_border=True):
        # 1: return ori img
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border

    def __call__(self, image, target=None):
        boxes = target['bboxes']
        w, h = image.size
        while True:
            mode = nprandom.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return image, target

            min_iou = mode
            for i in range(50):
                new_w = nprandom.uniform(self.min_crop_size * w, w)
                new_h = nprandom.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = nprandom.uniform(w - new_w)
                top = nprandom.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps_np(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue
                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = ((center[:, 0] > patch[0]) *
                                (center[:, 1] > patch[1]) *
                                (center[:, 0] < patch[2]) *
                                (center[:, 1] < patch[3]))
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    boxes = boxes[mask]
                    if self.bbox_clip_border:
                        boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                        boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                    boxes -= np.tile(patch[:2], 2)
                    target['bboxes'] = boxes
                    # labels
                    target['labels'] = target['labels'][mask]

                # adjust the img no matter whether the gt is empty before crop
                image_crop = image.crop(patch)
                target['img_size'] = image_crop.size

                return image_crop, target


@TRANSFORMS.register_module()
class Expand:
    def __init__(self,
                 mean=(0, 0, 0),
                 ratio_range=(1, 4),
                 prob=0.5):
        self.ratio_range = ratio_range
        self.mean = tuple([int(i) for i in mean])
        self.min_ratio, self.max_ratio = ratio_range
        self.prob = prob

    def __call__(self, image, target=None):
        if nprandom.uniform(0, 1) > self.prob:
            return image, target
        w, h = image.size
        ratio = nprandom.uniform(self.min_ratio, self.max_ratio)
        left = int(nprandom.uniform(0, w * ratio - w))
        top = int(nprandom.uniform(0, h * ratio - h))
        
        new_image = Image.new(image.mode,(int(w * ratio), int(h * ratio)), self.mean)
        new_image.paste(image,(left, top, left+image.size[0], top+image.size[1]))

        target["bboxes"] = target["bboxes"] + np.tile(
            (left, top), 2).astype(target["bboxes"].dtype)
        target['img_size'] = new_image.size
        return new_image, target

@TRANSFORMS.register_module()
class PhotoMetricDistortion:
    def __init__(self, brightness_delta=32./255, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.t = jt.transform.ColorJitter(brightness=brightness_delta, contrast=contrast_range, saturation=saturation_range, hue=hue_delta)

    def __call__(self, image, target=None):
        image = self.t(image)
        
        return image, target

@TRANSFORMS.register_module()
class Resize_keep_ratio:
    def __init__(self, min_size, max_size, keep_ratio=True):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.keep_ratio = keep_ratio

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size

        oh = self.min_size[0]
        ow = self.max_size
        return (oh, ow), [ow/w, oh/h, ow/w, oh/h]
        

    def _resize_boxes(self,target,size):
        for key in ["bboxes","polys"]:
            if key not in target:
                continue
            bboxes = target[key]
            width,height = target["img_size"]
            new_w,new_h = size
            bboxes[:,0::2] = bboxes[:,0::2]*float(new_w/width)
            bboxes[:,1::2] = bboxes[:,1::2]*float(new_h/height)

            # clip to border
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, new_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, new_h - 1)

            target[key]=bboxes
    
    def _resize_mask(self,target,size):
        pass

    def __call__(self, image, target=None):
        size,scale_factor = self.get_size(image.size)
        image = image.resize(size[::-1],Image.BILINEAR)
        if target is not None:
            self._resize_boxes(target,image.size)
            target["img_size"] = image.size
            target["scale_factor"] = scale_factor
            target["pad_shape"] = image.size
            target["keep_ratio"] = self.keep_ratio
        return image, target


@TRANSFORMS.register_module()
class RotatedResize(Resize):
    def __init__(self, min_size, max_size, angle_version='le135', keep_ratio=True):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.keep_ratio = keep_ratio
        self.angle_version = angle_version

    def _resize_boxes(self, target,size):
        for key in ["bboxes","hboxes","rboxes","polys","hboxes_ignore","polys_ignore","rboxes_ignore"]:
            if key not in target:
                continue
            bboxes = target[key]
            if bboxes is None or bboxes.ndim!=2:
                continue
            
            if "rboxes" in key:
                bboxes = rotated_box_to_poly_np(bboxes, self.angle_version)

            width,height = target["img_size"]
            new_w,new_h = size
            
            bboxes[:,0::2] = bboxes[:,0::2]*float(new_w/width)
            bboxes[:,1::2] = bboxes[:,1::2]*float(new_h/height)

            # clip to border
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, new_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, new_h - 1)
            
            if "rboxes" in key:
                bboxes = poly_to_rotated_box_np(bboxes, self.angle_version)
            target[key]=bboxes


@TRANSFORMS.register_module()
class RandomFlip:
    def __init__(self, prob=0.5,direction="horizontal"):
        assert direction in ['horizontal', 'vertical', 'diagonal'],f"{direction} not supported"
        self.direction = direction
        self.prob = prob
    
    def _flip_boxes(self,target,size):
        w,h = target["img_size"] 
        for key in ["bboxes","polys"]:
            if key not in target:
                continue
            bboxes = target[key]
            flipped = bboxes.copy()
            if self.direction == 'horizontal':
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
            elif self.direction == 'vertical':
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            elif self.direction == 'diagonal':
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            target[key] = flipped

    def _flip_image(self,image):
        if self.direction=="horizontal":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.direction == "vertical":
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif self.direction == "diagonal":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = self._flip_image(image)
            if target is not None:
                self._flip_boxes(target,image.size)
            target["flip"]=self.direction
        return image, target

@TRANSFORMS.register_module()
class RotatedRandomFlip(RandomFlip):
    def _flip_rboxes(self,bboxes,w,h):
        flipped = bboxes.copy()
        if self.direction == 'horizontal':
            flipped[..., 0::5] = w - flipped[..., 0::5] - 1
            flipped[..., 4::5] = norm_angle(np.pi - flipped[..., 4::5])
        elif self.direction == 'vertical':
            flipped[..., 1::5] = h - flipped[..., 1::5] - 1
            flipped[..., 4::5] = norm_angle( - flipped[..., 4::5])
        elif self.direction == 'diagonal':
            assert False
        else:
            assert False
        return flipped

    def _flip_polys(self,bboxes,w,h):
        flipped = bboxes.copy()
        if self.direction == 'horizontal':
            flipped[..., 0::2] = w - flipped[..., 0::2] - 1
        elif self.direction == 'vertical':
            flipped[..., 1::2] = h - flipped[..., 1::2] - 1
        elif self.direction == 'diagonal':
            flipped[..., 0::2] = w - flipped[..., 0::2] - 1
            flipped[..., 1::2] = h - flipped[..., 1::2] - 1
        return flipped 


    def _flip_boxes(self,target,size):
        w,h = size 
        for key in ["bboxes","hboxes","rboxes","polys","hboxes_ignore","polys_ignore","rboxes_ignore"]:
            if key not in target:
                continue
            bboxes = target[key]
            if "rboxes" in key:
                target[key] = self._flip_rboxes(bboxes,w,h)
                continue 
            if "polys" in key:
                target[key] = self._flip_polys(bboxes,w,h)
                continue
            flipped = bboxes.copy()
            if self.direction == 'horizontal':
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
            elif self.direction == 'vertical':
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            elif self.direction == 'diagonal':
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            target[key] = flipped

@TRANSFORMS.register_module()
class Pad:
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def __call__(self,image,target=None):
        if self.size is not None:
            pad_w,pad_h = self.size
        else:
            pad_h = int(np.ceil(image.size[1] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(image.size[0] / self.size_divisor)) * self.size_divisor
        
        new_image = Image.new(image.mode,(pad_w,pad_h),(self.pad_val,)*len(image.split()))
        new_image.paste(image,(0,0,image.size[0],image.size[1]))
        target["pad_shape"] = new_image.size
        
        return new_image,target
    

@TRANSFORMS.register_module()
class Normalize:
    def __init__(self, mean, std, to_bgr=True):
        self.mean = np.float32(mean).reshape(-1,1,1)
        self.std = np.float32(std).reshape(-1,1,1)
        self.to_bgr = to_bgr

    def __call__(self, image, target=None):
        if isinstance(image,Image.Image):
            image = np.array(image).transpose((2,0,1))

        if self.to_bgr:
            image = image[::-1]
        
        image = (image - self.mean) / self.std

        target["mean"] = self.mean 
        target["std"] = self.std
        target["to_bgr"] = self.to_bgr
        return image, target


@TRANSFORMS.register_module()
class YoloRandomPerspective:
    def __init__ (self, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0):
        self.degrees=degrees
        self.translate=translate
        self.scale=scale
        self.shear=shear
        self.perspective=perspective
    
    def __call__(self, img, targets=(), border=(0, 0)):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # targets = [cls, xyxy]

        height = img.shape[0] + border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # Visualize
        # import matplotlib.pyplot as plt
        # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
        # ax[0].imshow(img[:, :, ::-1])  # base
        # ax[1].imshow(img2[:, :, ::-1])  # warped

        # Transform label coordinates
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            if self.perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # # apply angle-based reduction of bounding boxes
            # radians = a * math.pi / 180
            # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            # x = (xy[:, 2] + xy[:, 0]) / 2
            # y = (xy[:, 3] + xy[:, 1]) / 2
            # w = (xy[:, 2] - xy[:, 0]) * reduction
            # h = (xy[:, 3] - xy[:, 1]) * reduction
            # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

            # filter candidates
            i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return img, targets

@TRANSFORMS.register_module()
class YoloAugmentHSV:
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain=hgain 
        self.sgain=sgain
        self.vgain=vgain 

    def __call__(self, img):
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


@TRANSFORMS.register_module()
class YoloRandomFlip:
    def __init__(self, prob=0.5, direction="horizontal"):
        assert direction in ['horizontal', 'vertical', 'diagonal'],f"{direction} not supported"
        self.direction = direction
        self.prob = prob

    def _flip_image(self, image):
        if self.direction=="horizontal":
            image = np.fliplr(image)
        elif self.direction=="vertical":
            image = np.flipud(image)
        elif self.direction == "diagonal":
            image = np.fliplr(image)
            image = np.flipud(image)
        return image

    def _flip_boxes(self, target):
        if self.direction=='horizontal':
            target[:, 1] = 1 - target[:, 1]
        elif self.direction=='vertical':
            labels[:, 2] = 1 - labels[:, 2]
        elif self.direction=='diagonal':
            target[:, 1] = 1 - target[:, 1]
            labels[:, 2] = 1 - labels[:, 2]
        
    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = self._flip_image(image)
            if target is not None:
                self._flip_boxes(target)
        return image, target

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates