import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from io import BytesIO
import torch
import lmdb


class LSUNBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example


class LSUNChurchesTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_train.txt", data_root="data/lsun/churches", **kwargs)


class LSUNChurchesValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_val.txt", data_root="data/lsun/churches",
                         flip_p=flip_p, **kwargs)


class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms",
                         flip_p=flip_p, **kwargs)


class LSUNCatsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)


class LSUNCatsValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats",
                         flip_p=flip_p, **kwargs)




def open_lmdb(path):
    env = lmdb.open(
        path,
        max_readers=32,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    return env.begin(write=False)

class HOUSE_LMDB(Dataset):
    def __init__(self, data_path, image_size, image_channel, flip_p=0.5):
        super().__init__()
        self.image_size = image_size
        self.image_channel = image_channel
        self.data_path = data_path

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return 3033042

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.txn = open_lmdb(self.data_path)

        key = f'256-{str(index).zfill(7)}'.encode('utf-8')
        img_bytes = self.txn.get(key)

        buffer = BytesIO(img_bytes)
        image = Image.open(buffer)

        image = self.transform(image).permute(1, 2, 0)
        
        return {
            "idx": index,
            "image": image
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        idx=[]
        x_0=[]
        for i in range(batch_size):
            idx.append(batch[i]["idx"])
            x_0.append(batch[i]["image"])

        x_0 = torch.stack(x_0, dim=0)

        return {
            "idx": idx,
            "image": x_0
        }


class BEDROOM_LMDB(Dataset):
    def __init__(self, data_path, image_size, image_channel, flip_p=0.5):
        super().__init__()
        self.image_size = image_size
        self.image_channel = image_channel
        self.data_path = data_path

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return 3033042

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.txn = open_lmdb(self.data_path)

        key = f'256-{str(index).zfill(7)}'.encode('utf-8')
        img_bytes = self.txn.get(key)

        buffer = BytesIO(img_bytes)
        image = Image.open(buffer)

        image = self.transform(image).permute(1, 2, 0)
        
        return {
            "idx": index,
            "image": image
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        idx=[]
        x_0=[]
        for i in range(batch_size):
            idx.append(batch[i]["idx"])
            x_0.append(batch[i]["image"])

        x_0 = torch.stack(x_0, dim=0)

        return {
            "idx": idx,
            "image": x_0
        }


class CHERCH_LMDB(Dataset):
    def __init__(self, data_path, image_size, image_channel, flip_p=0.5):
        super().__init__()
        self.image_size = image_size
        self.image_channel = image_channel
        self.data_path = data_path

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return 3033042

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.txn = open_lmdb(self.data_path)

        key = f'256-{str(index).zfill(7)}'.encode('utf-8')
        img_bytes = self.txn.get(key)

        buffer = BytesIO(img_bytes)
        image = Image.open(buffer)

        image = self.transform(image).permute(1, 2, 0)
        
        return {
            "idx": index,
            "image": image
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        idx=[]
        x_0=[]
        for i in range(batch_size):
            idx.append(batch[i]["idx"])
            x_0.append(batch[i]["image"])

        x_0 = torch.stack(x_0, dim=0)

        return {
            "idx": idx,
            "image": x_0
        }