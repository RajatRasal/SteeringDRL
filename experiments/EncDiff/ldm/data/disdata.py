from socket import SOL_IP
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import scipy.io as sio
from PIL import Image
from sklearn.utils.extmath import cartesian
import os
import lmdb
from io import BytesIO
import torchvision.transforms.functional as Ftran
import PIL
import glob


def cartesian_product(*arrays):
    ndim = len(arrays)
    return (np.stack(np.meshgrid(*arrays), axis=-1)
              .reshape(-1, ndim))
    
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15}
_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
# methods for sampling unconditionally/conditionally on a given factor
def get_index(factors):
    """ Converts factors to indices in range(num_data)
    Args:
    factors: np array shape [6,batch_size].
                factors[i]=factors[i,:] takes integer values in 
                range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

    Returns:
    indices: np array shape [batch_size].
    """
    indices = 0
    base = 1
    for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
        indices += factors[factor] * base
        base *= _NUM_VALUES_PER_FACTOR[name]
    return indices
class Shapes3D(Dataset):
    """
    also supports for d2c crop.
    """
    def __init__(self,
                 path,
                 original_resolution=64,
                 split=None,
                 as_tensor: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        # self.data = BaseLMDB(path, original_resolution, zfill=7)
        # data = np.load(os.path.join(path,'3dshapes.npz'))

        # self.data = data["images"]
        # self.length = len(self.data)

        import h5py
        self.hf = h5py.File(os.path.join(path, '3dshapes.h5'), "r")
        self.data = self.hf["images"][:]
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = []
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img).permute(1, 2, 0)
        return {'image': img,'idx' : index}


class Flower(Dataset):
    def __init__(self, path,
                original_resolution=64,
                split=None,
                as_tensor: bool = True,
                do_normalize: bool = True,
                **kwargs):
        super().__init__()
        self.original_resolution = original_resolution
        train_image_files = glob.glob(os.path.join(path,"train/*/*"))
        val_image_files = glob.glob(os.path.join(path,"valid/*/*"))
        test_image_files = glob.glob(os.path.join(path, "test/*/*"))

        if split is None:
            self.image_files = train_image_files + val_image_files + test_image_files
        elif split == 'val':
            self.image_files = val_image_files

        transform = []
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)
        self.interpolation = PIL.Image.BICUBIC


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.resize((self.original_resolution, self.original_resolution), resample=self.interpolation)
        if self.transform is not None:
            img = self.transform(image).permute(1, 2, 0)
        return {'image': img}


class AFHQ(Dataset):
    def __init__(self, path,
                original_resolution=64,
                split=None,
                as_tensor: bool = True,
                do_normalize: bool = True,
                **kwargs) -> None:
        super().__init__()
        self.original_resolution = original_resolution
        train_image_files = glob.glob(os.path.join(path,"train/*/*"))
        val_image_files = glob.glob(os.path.join(path,"val/*/*"))

        if split is None:
            self.image_files = train_image_files + val_image_files
        elif split == 'val':
            self.image_files = val_image_files


        transform = []
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)
        self.interpolation = PIL.Image.BICUBIC

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.resize((self.original_resolution, self.original_resolution), resample=self.interpolation)
        if self.transform is not None:
            img = self.transform(image).permute(1, 2, 0)
        return {'image': img}
        


class Shapes3D_SD(Dataset):
    """
    also supports for d2c crop.
    """
    def __init__(self,
                 path,
                 original_resolution=64,
                 split=None,
                 as_tensor: bool = True,
                 do_normalize: bool = True,
                 eval: bool = False,
                 **kwargs):
        self.original_resolution = original_resolution
        # self.data = BaseLMDB(path, original_resolution, zfill=7)
        data = np.load(os.path.join(path,'3dshapes.npz'))

        self.data = data["images"]
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = []
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)
        self.interpolation = PIL.Image.BICUBIC
        self.eval = eval

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if not self.eval:
            pil_img = Image.fromarray(img)
            # import pdb; pdb.set_trace()
            image_big = pil_img.resize((512, 512), resample=self.interpolation)
            if self.transform is not None:
                img = self.transform(img).permute(1, 2, 0)
            return {'image_small': img, 'image':self.transform(image_big).permute(1, 2, 0), 'idx' : index}
        else:
            if self.transform is not None:
                img = self.transform(img).permute(1, 2, 0)
            return {'image_small': img,'idx' : index}

class MPI3D(Dataset):
    """
    also supports for d2c crop.
    """
    def __init__(self,
                 path,
                 original_resolution=64,
                 split=None,
                 as_tensor: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        # self.data = BaseLMDB(path, original_resolution, zfill=7)
        data = np.load(os.path.join(path,"mpi3d_toy.npz"), "r")
        self.data = data["images"]
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = []
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img).permute(1, 2, 0)
        return {'image': img}

class MPI3D_SD(Dataset):
    """
    also supports for d2c crop.
    """
    def __init__(self,
                 path,
                 original_resolution=64,
                 split=None,
                 as_tensor: bool = True,
                 do_normalize: bool = True,
                 eval: bool = False,
                 **kwargs):
        self.original_resolution = original_resolution
        # self.data = BaseLMDB(path, original_resolution, zfill=7)
        data = np.load(os.path.join(path,"mpi_toy/mpi3d_toy.npz"), "r")
        self.data = data["images"]
        self.length = len(self.data)
        self.eval = eval

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = []
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)
        self.interpolation = PIL.Image.BICUBIC


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = Image.fromarray(self.data[index])
        if not self.eval:
            image_big = img.resize((512, 512), resample=self.interpolation)

            if self.transform is not None:
                img = self.transform(img).permute(1, 2, 0)
            return {'image_small': img, 'image':self.transform(image_big).permute(1, 2, 0), 'idx' : index}
        else:
            if self.transform is not None:
                img = self.transform(img).permute(1, 2, 0)
            return {'image_small': img, 'idx': index}   
def _features_to_state_space_index(features):
    """Returns the indices in the atom space for given factor configurations.
    Args:
    features: Numpy matrix where each row contains a different factor
        configuration for which the indices in the atom space should be
        returned.
    """
    factor_sizes = [4, 24, 183]
    num_total_atoms = np.prod(factor_sizes)
    factor_bases = num_total_atoms / np.cumprod(factor_sizes)
    if (np.any(features > np.expand_dims(factor_sizes, 0)) or
        np.any(features < 0)):
        raise ValueError("Feature indices have to be within [0, factor_size-1]!")
    return np.array(np.dot(features, factor_bases), dtype=np.int64)

def features_to_index(features):
    """Returns the indices in the input space for given factor configurations.
    Args:
        features: Numpy matrix where each row contains a different factor
        configuration for which the indices in the input space should be
        returned.
    """
    factor_sizes = [4, 24, 183]
    num_total_atoms = np.prod(factor_sizes)
    lookup_table = np.zeros(num_total_atoms, dtype=np.int64)
    global_features = cartesian([np.array(list(range(i))) for i in factor_sizes])
    feature_state_space_index = _features_to_state_space_index(global_features)
    lookup_table[feature_state_space_index] = np.arange(num_total_atoms)
    state_space_to_save_space_index = lookup_table
    state_space_index = _features_to_state_space_index(features)
    return state_space_to_save_space_index[state_space_index]

def _load_mesh(filename):
    """Parses a single source file and rescales contained images."""
    with open(filename, "rb") as f:
        mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
    flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
    rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
    for i in range(flattened_mesh.shape[0]):
        pic = Image.fromarray(flattened_mesh[i, :, :, :])
        pic.thumbnail((64, 64), Image.LANCZOS)
        rescaled_mesh[i, :, :, :] = np.array(pic)
    return rescaled_mesh * 1. / 255

def _load_data(dset_folder):
    dataset = np.zeros((24 * 4 * 183, 64, 64, 3))
    all_files = [x for x in os.listdir(dset_folder) if ".mat" in x]
    for i, filename in enumerate(all_files):
        data_mesh = _load_mesh(os.path.join(dset_folder, filename))
        factor1 = np.array(list(range(4)))
        factor2 = np.array(list(range(24)))
        all_factors = np.transpose([
          np.tile(factor1, len(factor2)),
          np.repeat(factor2, len(factor1)),
          np.tile(i,
                  len(factor1) * len(factor2))
        ])
        indexes = features_to_index(all_factors)
        dataset[indexes] = data_mesh
    return dataset

class Cars3D(Dataset):
    """
    also supports for d2c crop.
    """
    def __init__(self,
                 path,
                 original_resolution=64,
                 split=None,
                 as_tensor: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        # self.data = BaseLMDB(path, original_resolution, zfill=7)
        data = _load_data(os.path.join(path,"cars/"))
        if "test" not in kwargs.keys():
            self.data = np.repeat(np.uint8(data*255), 10, axis=0)
        else:
            self.data = np.uint8(data*255)
            # self.data = np.repeat(np.uint8(data*255), 20, axis=0)
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = []
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img).permute(1, 2, 0)
        return {'image': img}

class Clevr(Dataset):
    """
    also supports for d2c crop.
    """
    def __init__(self,
                 path,
                 original_resolution=64,
                 split=None,
                 as_tensor: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        # self.data = BaseLMDB(path, original_resolution, zfill=7)
        # path = os.path.join(path,"images_clevr/*.png")
        data = np.load(os.path.join(path,"clevr_npz/data.npz"), "r")
        self.data = data["imgs"]
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = []
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img).permute(1, 2, 0)
        return {'image': img}

class Crop:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return Ftrans.crop(img, self.x1, self.y1, self.x2 - self.x1,
                           self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2)

def d2c_crop():
    # from D2C paper for CelebA dataset.
    cx = 89
    cy = 121
    x1 = cy - 64
    x2 = cy + 64
    y1 = cx - 64
    y2 = cx + 64
    return Crop(x1, x2, y1, y2)

class BaseLMDB(Dataset):
    def __init__(self, path, original_resolution, zfill: int = 5):
        self.original_resolution = original_resolution
        self.zfill = zfill
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.original_resolution}-{str(index).zfill(self.zfill)}'.encode(
                'utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        return img
class CelebAlmdb(Dataset):
    """
    also supports for d2c crop.
    """
    def __init__(self,
                 path,
                 image_size,
                 original_resolution=128,
                 split=None,
                 as_tensor: bool = True,
                 do_augment: bool = False,
                 do_normalize: bool = True,
                 crop_d2c: bool = False,
                 **kwargs):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)
        self.crop_d2c = crop_d2c

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

        if crop_d2c:
            transform = [
                d2c_crop(),
                transforms.Resize(image_size),
            ]
        else:
            transform = [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]

        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img).permute(1, 2, 0)
        return {'image': img, 'idx':index}


class LSUNchurch(Dataset):
    def __init__(self,path,
                 original_resolution=64,
                 split=None,
                 as_tensor: bool = True,
                 do_normalize: bool = True,) -> None:
        super().__init__()
        self.data = np.load(os.path.join(path,"church_outdoor_train_lmdb_color_64.npy"))
        if split == 'val':
            self.data = self.data[:1000]
        self.original_resolution = original_resolution
        transform = []
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)
        self.interpolation = PIL.Image.BICUBIC
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        img = img.resize((self.original_resolution, self.original_resolution), resample=self.interpolation)
        if self.transform is not None:
            img = self.transform(img).permute(1, 2, 0)
        return {'image': img}


class Bedroom_lmdb(Dataset):
    def __init__(self,
                 path,
                 image_size=128,
                 original_resolution=256,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.data[index]
        img = self.transform(img).permute(1, 2, 0)
        return {'image': img}

class FFHQlmdb(Dataset):
    def __init__(self,
                 path,
                 image_size=256,
                 original_resolution=256,
                 split=None,
                 as_tensor: bool = True,
                 do_augment: bool = False,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=5)
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 10000
            self.offset = 10000
        elif split == 'test':
            # first 10k
            self.length = 10000
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img).permute(1, 2, 0)
        return {'image': img}

class Horse_lmdb(Dataset):
    def __init__(self,
                 path,
                 image_size=128,
                 original_resolution=256,
                 do_augment: bool = True,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        # print(path)
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img).permute(1, 2, 0)
        return {'image': img}
        
class Shapes3DTrain(Shapes3D):
    def __init__(self, **kwargs):
        super().__init__(path='/vol/biomedic2/rrr2417/Diffusion-Representation-Learning/datasets',
        # super().__init__(path='~/',
                original_resolution=None,
                **kwargs)

class Shapes3DTrainSD(Shapes3D_SD):
    def __init__(self, **kwargs):
        super().__init__(path='./datasets',
        # super().__init__(path='~/',
                original_resolution=None,
                **kwargs)

class MPI3DTrainSD(MPI3D_SD):
    def __init__(self, **kwargs):
        super().__init__(path='../../../guided-diffusion/datasets/',
        # super().__init__(path='~/',
                original_resolution=None,
                **kwargs)

class MPI3DTrain(MPI3D):
    def __init__(self, **kwargs):
        super().__init__(path='/vol/biomedic2/rrr2417/Diffusion-Representation-Learning/datasets',
                original_resolution=None,
                **kwargs)


class Cars3DTrain(Cars3D):
    def __init__(self, **kwargs):
        super().__init__(path='/vol/biomedic2/rrr2417/Diffusion-Representation-Learning/datasets/data/',
                original_resolution=None,
                **kwargs)

class ClevrTrain(Clevr):
    def __init__(self, **kwargs):
        super().__init__(path='../../../guided-diffusion/datasets/',
                original_resolution=None,
                **kwargs)

class Celebarain(CelebAlmdb):
    def __init__(self, **kwargs):
        super().__init__(path='../../diffusion_disen/diffae_new/datasets/celeba.lmdb/',
                image_size=64,
                original_resolution=None,
                crop_d2c=True,
                **kwargs)

class BedroomTrain(Bedroom_lmdb):
    def __init__(self, **kwargs):
        super().__init__(path='../../../guided-diffusion/datasets/bedroom256.lmdb/',
        # super().__init__(path='../diffae_new/datasets/celeba.lmdb/',
                image_size=64,
                original_resolution=256,
                # crop_d2c=True,
                **kwargs)

class HorseTrain(Horse_lmdb):
    def __init__(self, **kwargs):
        super().__init__(path='../../../guided-diffusion/datasets/horse256.lmdb/',
        # super().__init__(path='../diffae_new/datasets/celeba.lmdb/',
                image_size=64,
                original_resolution=256,
                # crop_d2c=True,
                **kwargs)

class FFHQTrain(FFHQlmdb):
    def __init__(self, **kwargs):
        super().__init__(path='../../../guided-diffusion/datasets/ffhq256.lmdb/',
                image_size=64,
                original_resolution=256,
                # crop_d2c=True,
                **kwargs)

class AFHQTrain(AFHQ):
    def __init__(self, **kwargs):
        super().__init__(path='/opt/tiger/ldm4pose/dataset/afhq',
                # original_resolution=64,
                # crop_d2c=True,
                **kwargs)

class AFHQVal(AFHQ):
    def __init__(self, **kwargs):
        super().__init__(path='/opt/tiger/ldm4pose/dataset/afhq',
                # original_resolution=64,
                split='val',
                # crop_d2c=True,
                **kwargs)

class FlowerTrain(Flower):
    def __init__(self, **kwargs):
        super().__init__(path='/opt/tiger/ldm4pose/dataset/flower_data',
                # original_resolution=64,
                # crop_d2c=True,
                **kwargs)
class FlowerVal(Flower):
    def __init__(self, **kwargs):
        super().__init__(path='/opt/tiger/ldm4pose/dataset/flower_data',
                # original_resolution=64,
                split='val',
                **kwargs)

class LSUNChurchTrain(LSUNchurch):
    def __init__(self, **kwargs):
        super().__init__(path='/opt/tiger/ldm4pose/dataset',
                # original_resolution=64,
                # crop_d2c=True,
                **kwargs)

class LSUNChurchVal(LSUNchurch):
    def __init__(self, **kwargs):
        super().__init__(path='/opt/tiger/ldm4pose/dataset',
                # original_resolution=64,
                # crop_d2c=True,
                split='val',
                **kwargs)
if __name__ == '__main__':
    church = LSUNChurchTrain()
    for batch in church:
        print(batch)
        import pdb; pdb.set_trace()