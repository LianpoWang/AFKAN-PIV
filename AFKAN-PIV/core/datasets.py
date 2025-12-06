import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
import math
import random
import re
from glob import glob
import os.path as osp

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from core.utils.unaugmentor import UnFlowAugmentor, UnSparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(** aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None

        flow = frame_utils.read_gen(self.flow_list[index])
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class UnFlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = UnSparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = UnFlowAugmentor(** aug_params)

        self.is_test = False
        self.init_seed = False
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            img1, img2 = self.augmentor(img1, img2)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        return img1, img2

    def __rmul__(self, v):
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class SimData(FlowDataset):
    def __init__(self, aug_params=None, split='testing', root='your_path'):
        super(SimData, self).__init__(aug_params)
        images1 = sorted(glob(osp.join(root, '*_img1.tif')))
        images2 = sorted(glob(osp.join(root, '*_img2.tif')))
        flow = sorted(glob(osp.join(root, '*_flow.flo')))

        for img1, img2, flow in zip(images1, images2, flow):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
            self.flow_list += [flow]


class Simdata(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='your_path/'):
        super(Simdata, self).__init__(aug_params)
        data_dir = osp.join(root, split)

        if not osp.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist")

        all_images1 = []
        all_images2 = []
        all_flow = []

        for sub_dir in os.listdir(data_dir):
            sub_dir_path = osp.join(data_dir, sub_dir)
            if os.path.isdir(sub_dir_path):
                images1 = sorted(glob(osp.join(sub_dir_path, '*_img1.tif')))
                images2 = sorted(glob(osp.join(sub_dir_path, '*_img2.tif')))
                flow = sorted(glob(osp.join(sub_dir_path, '*flow.flo')))

                if len(images1) == len(images2) == len(flow):
                    all_images1.extend(images1)
                    all_images2.extend(images2)
                    all_flow.extend(flow)
                else:
                    print(f"Warning: Skipping directory {sub_dir_path} due to mismatched file counts")

        if not all_images1 or not all_images2 or not all_flow:
            raise ValueError(f"No data found in {data_dir}")

        for img1, img2, flow in zip(all_images1, all_images2, all_flow):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
            self.flow_list += [flow]


class Wallshear(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='your_path'):
        super(Wallshear, self).__init__(aug_params)
        data_dir = osp.join(root, split)

        if not osp.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist")

        all_images1 = []
        all_images2 = []
        all_flow = []

        for sub_dir in os.listdir(data_dir):
            sub_dir_path = osp.join(data_dir, sub_dir)
            if os.path.isdir(sub_dir_path):
                images1 = sorted(glob(osp.join(sub_dir_path, '*_A.tif')))
                images2 = sorted(glob(osp.join(sub_dir_path, '*_B.tif')))
                flow = sorted(glob(osp.join(sub_dir_path, '*.flo')))

                if len(images1) == len(images2) == len(flow):
                    all_images1.extend(images1)
                    all_images2.extend(images2)
                    all_flow.extend(flow)
                else:
                    print(f"Warning: Skipping directory {sub_dir_path} due to mismatched file counts")

        if not all_images1 or not all_images2 or not all_flow:
            raise ValueError(f"No data found in {data_dir}")

        for img1, img2, flow in zip(all_images1, all_images2, all_flow):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
            self.flow_list += [flow]


class NestedMixedFlowDataset(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='your_path'):
        super(NestedMixedFlowDataset, self).__init__(aug_params)

        # 定义两种文件格式的匹配模式（根据实际文件命名调整）
        patterns = [
            {
                'img1': '*_img1.tif',
                'img2': '*_img2.tif',
                'flow': '*flow.flo'  # 匹配Simdata文件夹下的文件格式
            },
            {
                'img1': '*_A.tif',
                'img2': '*_B.tif',
                'flow': '*.flo'  # 匹配PAPER1_SIM文件夹下的文件格式
            }
        ]

        all_images1 = []
        all_images2 = []
        all_flow = []

        # 遍历主目录下的两个子文件夹（PAPER1_SIM和Simdata）
        for sub_root in os.listdir(root):
            sub_root_path = osp.join(root, sub_root)
            # 只处理PAPER1_SIM和Simdata这两个文件夹
            if not os.path.isdir(sub_root_path) or sub_root not in ['PAPER1_SIM', 'Simdata']:
                continue

            # 关键修改：进入当前子文件夹下的split目录（train或test）
            split_dir = osp.join(sub_root_path, split)  # 例如：PAPER1_SIM/train 或 Simdata/test
            if not osp.exists(split_dir):
                print(f"Warning: {split} directory not found in {sub_root_path}, skipping this folder")
                continue

            # 遍历split目录下的所有流场文件夹（数据实际存放的子目录）
            for flow_field_dir in os.listdir(split_dir):
                flow_field_path = osp.join(split_dir, flow_field_dir)
                if not os.path.isdir(flow_field_path):
                    continue  # 只处理流场文件夹，跳过文件

                found = False
                # 尝试两种格式匹配当前流场文件夹下的数据
                for pattern in patterns:
                    # 查找符合当前模式的文件
                    images1 = sorted(glob(osp.join(flow_field_path, pattern['img1'])))
                    images2 = sorted(glob(osp.join(flow_field_path, pattern['img2'])))
                    flow_files = sorted(glob(osp.join(flow_field_path, pattern['flow'])))

                    # 检查文件数量是否匹配且不为空
                    if len(images1) == len(images2) == len(flow_files) and len(images1) > 0:
                        all_images1.extend(images1)
                        all_images2.extend(images2)
                        all_flow.extend(flow_files)
                        found = True
                        print(
                            f"Loaded {len(images1)} pairs from {flow_field_path} (matched pattern: {pattern['img1']})")
                        break  # 匹配到一种格式后停止尝试其他模式

                if not found:
                    print(f"Warning: No valid file pattern matched in {flow_field_path}, skipping this flow field folder")

        # 检查是否加载到数据
        if not all_images1 or not all_images2 or not all_flow:
            raise ValueError(f"No valid data found in {split} directories under {root}")

        # 确保三类文件数量一致
        if not (len(all_images1) == len(all_images2) == len(all_flow)):
            raise ValueError(
                f"Mismatched data counts: images1={len(all_images1)}, images2={len(all_images2)}, flow={len(all_flow)}")

        # 加载数据到列表
        for img1, img2, flow in zip(all_images1, all_images2, all_flow):
            frame_id = osp.basename(img1)
            self.extra_info.append([frame_id])
            self.image_list.append([img1, img2])
            self.flow_list.append(flow)

        print(f"Successfully loaded total {len(all_images1)} data pairs (split: {split}) from nested directories")


def fetch_dataloader(args):
    """ 创建训练集和验证集的数据加载器（适配NestedMixedFlowDataset） """
    aug_params = {
        'crop_size': args.image_size,
        'min_scale': 0,
        'max_scale': 0,
        'do_flip': args.do_flip
    }

    # 训练数据集：使用主目录下的train split
    train_dataset = Wallshear(
        aug_params=aug_params,
        split='train',  # 明确指定加载train目录
        root=args.test_data_root
    )

    # 验证/测试数据集：使用主目录下的test split
    val_dataset = Wallshear(
        aug_params=None,  # 测试不增强
        split='test',  # 明确指定加载test目录
        root=args.test_data_root
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    print(f"Training with {len(train_dataset)} image pairs")
    print(f"Validation with {len(val_dataset)} image pairs")

    return train_loader, val_loader