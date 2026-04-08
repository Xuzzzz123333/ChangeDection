from .transform import Transforms
import numpy as np
import os
import sys
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


def make_dataset(dir):
    img_paths = []
    names = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            img_paths.append(path)
            names.append(fname)

    return img_paths, names


def import_custom_change_dataset():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from dataset import ChangeDataset
    except ImportError as exc:
        raise ImportError(
            "Failed to import the custom ChangeDataset from the project root. "
            "Make sure dataset.py exists alongside the ChangeDINO folder."
        ) from exc

    return ChangeDataset


def resolve_custom_dataset_root(opt):
    candidates = []
    joined_root = os.path.join(opt.dataroot, opt.dataset)
    candidates.append(joined_root)
    if os.path.abspath(opt.dataroot) != os.path.abspath(joined_root):
        candidates.append(opt.dataroot)

    required_relpaths = (
        os.path.join(opt.phase, "A"),
        os.path.join(opt.phase, "B"),
        os.path.join(opt.phase, "label"),
    )

    for candidate in candidates:
        if all(os.path.isdir(os.path.join(candidate, relpath)) for relpath in required_relpaths):
            return candidate

    expected = ", ".join(os.path.join(joined_root, relpath) for relpath in required_relpaths)
    raise FileNotFoundError(
        "Could not locate a valid custom_patch dataset root. "
        f"Tried: {candidates}. Expected to find: {expected}"
    )


class Load_Dataset(Dataset):
    def __init__(self, opt):
        super(Load_Dataset, self).__init__()
        self.opt = opt
        self.data_mode = getattr(opt, "data_mode", "original")
        self.eval_size = (256, 256)
        self.normalize_mean = (0.430, 0.411, 0.296)
        self.normalize_std = (0.213, 0.156, 0.143)

        if self.data_mode == "custom_patch":
            dataset_root = resolve_custom_dataset_root(opt)
            ChangeDataset = import_custom_change_dataset()
            self.custom_dataset = ChangeDataset(
                root=dataset_root,
                split=opt.phase,
                image_size=opt.image_size,
                normalize_mean=self.normalize_mean,
                normalize_std=self.normalize_std,
            )
            self.dataset_size = len(self.custom_dataset)
            if self.dataset_size <= 0:
                raise ValueError(
                    "The custom_patch dataset is empty. "
                    f"Resolved root: {dataset_root}, split: {opt.phase}"
                )
            self.transform = None
            self.normalize = None
            return

        self.dir1 = os.path.join(opt.dataroot, opt.dataset, opt.phase, "A")
        self.t1_paths, self.fnames = sorted(make_dataset(self.dir1))

        self.dir2 = os.path.join(opt.dataroot, opt.dataset, opt.phase, "B")
        self.t2_paths, _ = sorted(make_dataset(self.dir2))

        self.dir_label = os.path.join(opt.dataroot, opt.dataset, opt.phase, "label")
        self.label_paths, _ = sorted(make_dataset(self.dir_label))

        self.dataset_size = len(self.t1_paths)

        self.normalize = transforms.Compose(
            [transforms.Normalize(self.normalize_mean, self.normalize_std)]
        )
        self.transform = transforms.Compose([Transforms()])

    def _image_to_tensor(self, img):
        arr = np.array(img, copy=True)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        tensor = torch.tensor(arr, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1).contiguous() / 255.0
        return tensor

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        if self.data_mode == "custom_patch":
            sample = self.custom_dataset[index]
            return {
                "img1": sample["img1"].contiguous(),
                "img2": sample["img2"].contiguous(),
                "cd_label": sample["mask"].squeeze(0).to(dtype=torch.long).contiguous(),
                "fname": sample["name"],
            }

        t1_path = self.t1_paths[index]
        fname = self.fnames[index]
        img1 = Image.open(t1_path)

        t2_path = self.t2_paths[index]
        img2 = Image.open(t2_path)

        label_path = self.label_paths[index]
        label = np.array(Image.open(label_path)) / 255
        label[label > 0] = 1
        cd_label = Image.fromarray(label)

        if self.opt.phase == "train":
            _data = self.transform({"img1": img1, "img2": img2, "cd_label": cd_label})
            img1, img2, cd_label = _data["img1"], _data["img2"], _data["cd_label"]
        else:
            img1 = TF.resize(img1, size=self.eval_size, interpolation=InterpolationMode.BILINEAR)
            img2 = TF.resize(img2, size=self.eval_size, interpolation=InterpolationMode.BILINEAR)
            cd_label = TF.resize(cd_label, size=self.eval_size, interpolation=InterpolationMode.NEAREST)

        img1 = self._image_to_tensor(img1)
        img2 = self._image_to_tensor(img2)
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        cd_label = torch.tensor(np.array(cd_label, copy=True), dtype=torch.long)
        input_dict = {"img1": img1, "img2": img2, "cd_label": cd_label, "fname": fname}

        return input_dict


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, opt):
        self.dataset = Load_Dataset(opt)
        self.sampler = None
        if getattr(opt, "distributed", False):
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=opt.world_size,
                rank=opt.rank,
                shuffle=opt.phase == "train",
                drop_last=opt.phase == "train",
            )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=self.sampler is None and opt.phase == "train",
            sampler=self.sampler,
            pin_memory=True,
            drop_last=opt.phase == "train",
            num_workers=int(opt.num_workers),
            collate_fn=self._collate_fn,
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    @staticmethod
    def _collate_fn(batch):
        return {
            "img1": torch.stack([item["img1"].clone() for item in batch], dim=0),
            "img2": torch.stack([item["img2"].clone() for item in batch], dim=0),
            "cd_label": torch.stack([item["cd_label"].clone() for item in batch], dim=0),
            "fname": [item["fname"] for item in batch],
        }
