from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
import cv2
# from enhance.process_ref import ProcssRef

import monai
from monai.transforms import Compose
from monai.data import DataLoader, CacheDataset

from torchvision.transforms import Lambda
import torch.distributed as dist
from setting import partition_dataset
import random

# --- 新增：临床退化模拟工具类 ---
class ClinicalDegradation:
    """
    更真实的临床 MRI 退化增强版本：
    - 轻度运动模糊
    - 轻中度噪声（Rician）
    - 合理偏置场（Bias Field）
    - 小范围错位（Misalignment）

    所有退化都保证：保持结构仍可辨认，符合真实临床情况
    """

    def __init__(self):
        pass

    # ------------------------------------------------------
    # 1. Motion Blur (轻度)
    # ------------------------------------------------------
    def apply_motion_blur(self, img_np):
        """真实临床友好的轻度运动模糊"""
        kernel_size = random.choice([3, 5, 7])  # 临床轻模糊范围
        angle = random.uniform(-10, 10)

        kernel = np.zeros((kernel_size, kernel_size))
        np.fill_diagonal(kernel, 1)

        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

        # 加入强度权重（避免模糊全图）
        strength = random.uniform(0.1, 0.4)
        kernel = kernel * strength

        kernel = kernel / np.sum(kernel)

        blurred = cv2.filter2D(img_np, -1, kernel)
        return blurred

    # ------------------------------------------------------
    # 2. Bias Field (真实低频)
    # ------------------------------------------------------
    def apply_bias_field(self, img_np):
        """更合理的偏置场（强度不均匀）"""

        h, w = img_np.shape[:2]
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)

        scale = random.uniform(0.05, 0.15)  # 临床真实范围 5%-15%

        bias = 1 + scale * (X**2 + Y**2)
        bias = bias.astype(np.float32)

        img_new = img_np.astype(np.float32) * bias
        return np.clip(img_new, 0, 255).astype(np.uint8)

    # ------------------------------------------------------
    # 3. Misalignment（小范围）
    # ------------------------------------------------------
    def apply_misalignment(self, img_np):
        """轻度配准误差"""
        h, w = img_np.shape[:2]

        max_shift = 1.5          # 临床级 shift 范围
        max_angle = 1.0          # 几乎不发生大旋转

        tx = random.uniform(-max_shift, max_shift)
        ty = random.uniform(-max_shift, max_shift)
        angle = random.uniform(-max_angle, max_angle)

        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        M[0, 2] += tx
        M[1, 2] += ty

        warped = cv2.warpAffine(img_np, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return warped

    # ------------------------------------------------------
    # 4. Rician Noise（真实噪声水平）
    # ------------------------------------------------------
    def apply_rician_noise(self, img_np):
        """真实水平的 Rician Noise，保持结构可辨认"""
        sigma = random.uniform(2, 6)     # 文献中真实 MRI 噪声范围
        snr_scale = random.uniform(0.4, 1.0)

        noise1 = np.random.normal(0, sigma, img_np.shape)
        noise2 = np.random.normal(0, sigma, img_np.shape)

        img_float = img_np.astype(np.float32)

        noisy = np.sqrt((img_float + noise1)**2 + noise2**2)
        noisy = noisy * snr_scale

        return np.clip(noisy, 0, 255).astype(np.uint8)
    
class BraTs_datasets(Dataset):
    def __init__(self, path_Data, config, train=True, use_clinical_degradation=None):
        super(BraTs_datasets, self)
        self.train = train
        self.config = config
        self.degrader = ClinicalDegradation()

        # ---------------------------------
        # 临床退化增强开关
        # 优先使用显式传参，否则从 config 里读取
        # 若 config 中没有该属性，则默认开启训练时退化增强
        # ---------------------------------
        if use_clinical_degradation is None:
            self.use_clinical_degradation = getattr(
                config, "use_clinical_degradation", True
            )
        else:
            self.use_clinical_degradation = use_clinical_degradation

        # -------------------------
        # 选择不同数据集的文件夹名称
        # -------------------------
        if config.datasets in ['BraTs2020_t1_t2_small', 'Private_kidney']:
            # === 默认保持你的原始设置 ===
            ori_folder = 'oriT2/'
            ref_folder = 'oriT1/'
            ref_lr_folder = 'orLRbicT1/'
            tar_folder = 'orLRbicT2/'

        elif config.datasets == 'IXI_t1_t2_tiny':
            # === IXI: T2 target，PD reference ===
            ori_folder = 'oriT2/'
            ref_folder = 'oriPD/'
            ref_lr_folder = 'orLRbicPD/'
            tar_folder = 'orLRbicT2/'

        elif config.datasets == 'fastmri_tiny':
            # === fastMRI: PD target，FSPD reference ===
            ori_folder = 'oriFSPD/'
            ref_folder = 'oriPD/'
            ref_lr_folder = 'orLRbicPD/'
            tar_folder = 'orLRbicFSPD/'

        else:
            raise ValueError(f"Unknown dataset type: {config.datasets}")

        # -------------------------
        # 根据 train / val 选择路径
        # -------------------------
        phase = 'train' if train else 'val'

        ori_list = sorted(os.listdir(f"{path_Data}{phase}/{ori_folder}"))
        ref_list = sorted(os.listdir(f"{path_Data}{phase}/{ref_folder}"))
        ref_lr_list = sorted(os.listdir(f"{path_Data}{phase}/{ref_lr_folder}/x{config.upscale}/"))
        tar_list = sorted(os.listdir(f"{path_Data}{phase}/{tar_folder}/x{config.upscale}/"))

        # -------------------------
        # 保存数据路径
        # -------------------------
        self.data = []
        for i in range(len(ori_list)):
            img_name = ori_list[i]

            ori_path = f"{path_Data}{phase}/{ori_folder}{ori_list[i]}"
            ref_path = f"{path_Data}{phase}/{ref_folder}{ref_list[i]}"
            ref_lr_path = f"{path_Data}{phase}/{ref_lr_folder}/x{config.upscale}/{ref_lr_list[i]}"
            tar_path = f"{path_Data}{phase}/{tar_folder}/x{config.upscale}/{tar_list[i]}"

            self.data.append([ori_path, ref_path, ref_lr_path, tar_path, img_name])

        self.transformer = config.train_transformer if train else config.test_transformer

    def __getitem__(self, indx):
        ori_path, ref_path, ref_lr_path, tar_path, img_name = self.data[indx]

        ori = np.array(Image.open(ori_path).convert('RGB'))
        ref = np.array(Image.open(ref_path).convert('RGB'))
        ref_lr = np.array(Image.open(ref_lr_path).convert('RGB'))
        tar = np.array(Image.open(tar_path).convert('RGB'))

        # ------------------------------
        # 训练模式 + 开关开启时：加入临床退化增强
        # ------------------------------
        if self.train and self.use_clinical_degradation:

            # 20% 概率添加 bias field
            if random.random() < 0.2:
                tar = self.degrader.apply_bias_field(tar)
                ref = self.degrader.apply_bias_field(ref)

            # 30% 概率让 ref 降质（运动模糊 / misalignment）
            if random.random() < 0.3:
                if random.random() < 0.5:
                    ref = self.degrader.apply_motion_blur(ref, angle=random.randint(0, 90))
                else:
                    ref = self.degrader.apply_misalignment(ref)

            # 20% 概率让 tar 增加 Rician 噪声
            elif random.random() < 0.2:
                tar = self.degrader.apply_rician_noise(tar, sigma=15)

        # -------------------------
        # 转为 PIL 用 transforms
        # -------------------------
        ori = self.transformer(Image.fromarray(ori))
        ref = self.transformer(Image.fromarray(ref))
        ref_lr = self.transformer(Image.fromarray(ref_lr))
        tar = self.transformer(Image.fromarray(tar))

        return ori, ref, ref_lr, tar, img_name

    def __len__(self):
        return len(self.data)



def prepare_data(data, config, device, cache_rate, batch_size, num_workers=0, train=True):
    """
    Prepare training data.

    Args:
        train_files (list): List of training files.
        device (torch.device): Device to use for training.
        cache_rate (float): Cache rate for dataset.
        num_workers (int): Number of workers for data loading.
        batch_size (int): Mini-batch size.

    Returns:
        DataLoader: Data loader for training.
    """
    if train:
        train_transforms = Compose(
        [
            monai.transforms.LoadImaged(keys=["ori_path"]),
            monai.transforms.EnsureChannelFirstd(keys=["ori_path"]),
            monai.transforms.RepeatChanneld(keys=["ori_path"], repeats=3),
            monai.transforms.NormalizeIntensityd(keys=["ori_path"]),
            monai.transforms.LoadImaged(keys=["ref_path"]),
            monai.transforms.EnsureChannelFirstd(keys=["ref_path"]),
            monai.transforms.RepeatChanneld(keys=["ref_path"], repeats=3),
            monai.transforms.NormalizeIntensityd(keys=["ref_path"]),
            monai.transforms.LoadImaged(keys=["ref_lr_path"]),
            monai.transforms.EnsureChannelFirstd(keys=["ref_lr_path"]),
            monai.transforms.NormalizeIntensityd(keys=["ref_lr_path"]),
            monai.transforms.LoadImaged(keys=["tar_path"]),
            monai.transforms.EnsureChannelFirstd(keys=["tar_path"]),
            monai.transforms.NormalizeIntensityd(keys=["tar_path"]),
        ]
        )
        infer_ds = CacheDataset(
        data=data, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
        )
    else:
        test_transforms = Compose(
        [
            monai.transforms.LoadImaged(keys=["ori_path"]),
            monai.transforms.EnsureChannelFirstd(keys=["ori_path"]),
            monai.transforms.RepeatChanneld(keys=["ori_path"], repeats=3),
            monai.transforms.NormalizeIntensityd(keys=["ori_path"]),
            monai.transforms.LoadImaged(keys=["ref_path"]),
            monai.transforms.EnsureChannelFirstd(keys=["ref_path"]),
            monai.transforms.RepeatChanneld(keys=["ref_path"], repeats=3),
            monai.transforms.NormalizeIntensityd(keys=["ref_path"]),
            monai.transforms.LoadImaged(keys=["ref_lr_path"]),
            monai.transforms.EnsureChannelFirstd(keys=["ref_lr_path"]),
            monai.transforms.NormalizeIntensityd(keys=["ref_lr_path"]),
            monai.transforms.LoadImaged(keys=["tar_path"]),
            monai.transforms.EnsureChannelFirstd(keys=["tar_path"]),
            monai.transforms.NormalizeIntensityd(keys=["tar_path"]),
        ]
        )
        infer_ds = CacheDataset(
        data=data, transform=test_transforms, cache_rate=cache_rate, num_workers=num_workers
        )
    

    return DataLoader(infer_ds, num_workers=0, batch_size=batch_size, shuffle=True)

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
        
    
