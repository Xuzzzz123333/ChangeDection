import math
import random

import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class TransformsV2(object):
    def __init__(self, opt):
        self.image_size = int(opt.image_size)
        self.temporal_swap_enable = bool(getattr(opt, "temporal_swap_enable", False))
        self.asym_color_jitter_enable = bool(
            getattr(opt, "asym_color_jitter_enable", False)
        )
        self.asym_color_jitter_prob = float(
            getattr(opt, "asym_color_jitter_prob", 0.5)
        )
        self.asym_brightness = float(getattr(opt, "asym_brightness", 0.2))
        self.asym_contrast = float(getattr(opt, "asym_contrast", 0.2))
        self.asym_saturation = float(getattr(opt, "asym_saturation", 0.2))
        self.asym_hue = float(getattr(opt, "asym_hue", 0.05))
        self.change_aware_crop_enable = bool(
            getattr(opt, "change_aware_crop_enable", False)
        )
        self.change_aware_crop_prob = float(
            getattr(opt, "change_aware_crop_prob", 0.5)
        )
        self.change_aware_crop_min_ratio = float(
            getattr(opt, "change_aware_crop_min_ratio", 0.005)
        )

    def __call__(self, _data):
        img1, img2, cd_label = _data["img1"], _data["img2"], _data["cd_label"]

        if self.temporal_swap_enable and random.random() < 0.5:
            img1, img2 = img2, img1

        if random.random() < 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            cd_label = TF.hflip(cd_label)

        if random.random() < 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            cd_label = TF.vflip(cd_label)

        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            img1 = TF.rotate(img1, angle)
            img2 = TF.rotate(img2, angle)
            cd_label = TF.rotate(cd_label, angle)

        if self.asym_color_jitter_enable:
            if random.random() < self.asym_color_jitter_prob:
                img1 = self._apply_asym_color_jitter(img1)
                img2 = self._apply_asym_color_jitter(img2)
        elif random.random() < 0.5:
            colorjitters = []
            brightness_factor = random.uniform(0.75, 1.25)
            colorjitters.append(
                Lambda(lambda img: TF.adjust_brightness(img, brightness_factor))
            )
            contrast_factor = random.uniform(0.75, 1.25)
            colorjitters.append(
                Lambda(lambda img: TF.adjust_contrast(img, contrast_factor))
            )
            saturation_factor = random.uniform(0.75, 1.25)
            colorjitters.append(
                Lambda(lambda img: TF.adjust_saturation(img, saturation_factor))
            )
            random.shuffle(colorjitters)
            colorjitter = Compose(colorjitters)
            img1 = colorjitter(img1)
            img2 = colorjitter(img2)

        cropped = False
        if (
            self.change_aware_crop_enable
            and random.random() < self.change_aware_crop_prob
            and self._change_ratio(cd_label) >= self.change_aware_crop_min_ratio
        ):
            crop_box = self._get_change_aware_crop_params(cd_label)
            if crop_box is not None:
                i, j, h, w = crop_box
                img1 = TF.resized_crop(
                    img1,
                    i,
                    j,
                    h,
                    w,
                    size=(self.image_size, self.image_size),
                    interpolation=InterpolationMode.BILINEAR,
                )
                img2 = TF.resized_crop(
                    img2,
                    i,
                    j,
                    h,
                    w,
                    size=(self.image_size, self.image_size),
                    interpolation=InterpolationMode.BILINEAR,
                )
                cd_label = TF.resized_crop(
                    cd_label,
                    i,
                    j,
                    h,
                    w,
                    size=(self.image_size, self.image_size),
                    interpolation=InterpolationMode.NEAREST,
                )
                cropped = True

        if not cropped and random.random() < 0.5:
            i, j, h, w = transforms.RandomResizedCrop(
                size=(self.image_size, self.image_size)
            ).get_params(img=img1, scale=[0.333, 1.0], ratio=[0.75, 1.333])
            img1 = TF.resized_crop(
                img1,
                i,
                j,
                h,
                w,
                size=(self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR,
            )
            img2 = TF.resized_crop(
                img2,
                i,
                j,
                h,
                w,
                size=(self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR,
            )
            cd_label = TF.resized_crop(
                cd_label,
                i,
                j,
                h,
                w,
                size=(self.image_size, self.image_size),
                interpolation=InterpolationMode.NEAREST,
            )
            cropped = True

        if not cropped:
            img1 = TF.resize(
                img1,
                size=(self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR,
            )
            img2 = TF.resize(
                img2,
                size=(self.image_size, self.image_size),
                interpolation=InterpolationMode.BILINEAR,
            )
            cd_label = TF.resize(
                cd_label,
                size=(self.image_size, self.image_size),
                interpolation=InterpolationMode.NEAREST,
            )

        return {"img1": img1, "img2": img2, "cd_label": cd_label}

    def _apply_asym_color_jitter(self, img):
        colorjitters = []
        brightness_factor = random.uniform(
            max(0.0, 1.0 - self.asym_brightness), 1.0 + self.asym_brightness
        )
        colorjitters.append(
            Lambda(lambda x: TF.adjust_brightness(x, brightness_factor))
        )
        contrast_factor = random.uniform(
            max(0.0, 1.0 - self.asym_contrast), 1.0 + self.asym_contrast
        )
        colorjitters.append(Lambda(lambda x: TF.adjust_contrast(x, contrast_factor)))
        saturation_factor = random.uniform(
            max(0.0, 1.0 - self.asym_saturation), 1.0 + self.asym_saturation
        )
        colorjitters.append(
            Lambda(lambda x: TF.adjust_saturation(x, saturation_factor))
        )
        if self.asym_hue > 0:
            hue_factor = random.uniform(-self.asym_hue, self.asym_hue)
            colorjitters.append(Lambda(lambda x: TF.adjust_hue(x, hue_factor)))
        random.shuffle(colorjitters)
        return Compose(colorjitters)(img)

    def _change_ratio(self, cd_label):
        label_np = np.array(cd_label, copy=False)
        if label_np.size == 0:
            return 0.0
        return float((label_np > 0).mean())

    def _get_change_aware_crop_params(self, cd_label):
        label_np = np.array(cd_label, copy=False)
        change_coords = np.argwhere(label_np > 0)
        if change_coords.size == 0:
            return None

        height, width = label_np.shape[:2]
        area = height * width
        log_ratio = (math.log(0.75), math.log(1.333))

        for _ in range(10):
            target_area = area * random.uniform(0.333, 1.0)
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                y, x = change_coords[random.randrange(len(change_coords))]
                i_min = max(0, int(y) - h + 1)
                i_max = min(int(y), height - h)
                j_min = max(0, int(x) - w + 1)
                j_max = min(int(x), width - w)
                if i_min <= i_max and j_min <= j_max:
                    i = random.randint(i_min, i_max)
                    j = random.randint(j_min, j_max)
                    return i, j, h, w

        in_ratio = float(width) / float(height)
        if in_ratio < 0.75:
            w = width
            h = int(round(w / 0.75))
        elif in_ratio > 1.333:
            h = height
            w = int(round(h * 1.333))
        else:
            w = width
            h = height

        y, x = change_coords[random.randrange(len(change_coords))]
        i_min = max(0, int(y) - h + 1)
        i_max = min(int(y), height - h)
        j_min = max(0, int(x) - w + 1)
        j_max = min(int(x), width - w)
        if i_min <= i_max and j_min <= j_max:
            i = random.randint(i_min, i_max)
            j = random.randint(j_min, j_max)
            return i, j, h, w
        return None


class Lambda(object):
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
