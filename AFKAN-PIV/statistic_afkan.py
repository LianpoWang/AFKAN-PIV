from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from model.AFKAN_PIV.afkan_piv import AFKAN_PIV
import tqdm
import argparse
import os
import math
import random
import re
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os.path as osp
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

# Utilities for datasets
from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def split_image_into_blocks(img, block_size, overlap):
    """Split image into overlapping blocks"""
    C, H, W = img.shape
    block_h, block_w = block_size
    blocks = []
    coords = []  # Record coordinates of each block in original image (start_h, end_h, start_w, end_w)

    # Stride = block size - overlap pixels
    step_h = block_h - overlap
    step_w = block_w - overlap

    # Calculate number of rows and columns (ensure full coverage)
    num_rows = max(1, (H - overlap + step_h - 1) // step_h)  # Ceil division
    num_cols = max(1, (W - overlap + step_w - 1) // step_w)

    for i in range(num_rows):
        for j in range(num_cols):
            # Calculate block coordinates
            start_h = i * step_h
            end_h = start_h + block_h
            start_w = j * step_w
            end_w = start_w + block_w

            # Boundary adjustment (avoid exceeding original image)
            if end_h > H:
                end_h = H
                start_h = max(0, end_h - block_h)
            if end_w > W:
                end_w = W
                start_w = max(0, end_w - block_w)

            # Extract block
            block = img[:, start_h:end_h, start_w:end_w]
            blocks.append(block)
            coords.append((start_h, end_h, start_w, end_w))

    return blocks, coords


def generate_weight_mask(block_h, block_w, overlap, device):
    """Generate block weight mask (high center weight, low edge weight, for merging overlapping areas)"""
    # Horizontal weights: linear increase at edges (overlap), 1 in center
    if block_h <= 2 * overlap:
        h_weight = torch.linspace(0.1, 1.0, block_h, device=device)
    else:
        h_edge = torch.linspace(0.1, 1.0, overlap, device=device)
        h_center = torch.ones(block_h - 2 * overlap, device=device)
        h_weight = torch.cat([h_edge, h_center, h_edge.flip(dims=[0])])

    # Vertical weights
    if block_w <= 2 * overlap:
        w_weight = torch.linspace(0.1, 1.0, block_w, device=device)
    else:
        w_edge = torch.linspace(0.1, 1.0, overlap, device=device)
        w_center = torch.ones(block_w - 2 * overlap, device=device)
        w_weight = torch.cat([w_edge, w_center, w_edge.flip(dims=[0])])

    # 2D weight mask (outer product)
    weight_mask = torch.outer(h_weight, w_weight)  # [block_h, block_w]
    return weight_mask


def merge_flow_blocks(flow_blocks, coords, original_shape, overlap, device):
    """Merge flow blocks, weighted fusion in overlapping areas"""
    H, W = original_shape
    full_flow = torch.zeros((2, H, W), device=device)  # Full flow field
    weight_sum = torch.zeros((H, W), device=device)  # Accumulated weights

    for flow_block, (start_h, end_h, start_w, end_w) in zip(flow_blocks, coords):
        block_h = end_h - start_h
        block_w = end_w - start_w

        # Generate weight mask for current block
        weight_mask = generate_weight_mask(block_h, block_w, overlap, device)  # [block_h, block_w]

        # Weighted accumulation of flow and weights
        full_flow[:, start_h:end_h, start_w:end_w] += flow_block * weight_mask
        weight_sum[start_h:end_h, start_w:end_w] += weight_mask

    # Divide by total weight to get final flow field (avoid division by zero)
    full_flow = full_flow / torch.clamp(weight_sum, min=1e-8)
    return full_flow

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

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

        # Grayscale to RGB
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

class Wallshear(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='Our_data_path'):
        super(Wallshear, self).__init__(aug_params)
        self.split = split
        self.root = root
        data_dir = osp.join(root, split)

        if not osp.exists(data_dir):
            raise ValueError(f"Dataset directory does not exist: {data_dir}")

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
                    print(f"Warning: Skipping directory {sub_dir_path} (file count mismatch)")

        if not all_images1 or not all_images2 or not all_flow:
            raise ValueError(f"No valid data found in {data_dir}")

        for img1, img2, flow in zip(all_images1, all_images2, all_flow):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
            self.flow_list += [flow]

class Simdata(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='Public_data_path'):
        super(Simdata, self).__init__(aug_params)
        self.split = split
        self.root = root
        data_dir = osp.join(root, split)
        self.flow_types = ["backstep", "cylinder", "DNS_turbulence", "JHTDB_channel", "JHTDB_channel_hd",
                           "JHTDB_isotropic1024_hd", "JHTDB_mhd1024_hd", "SQG", "uniform"]
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

class Lagemann(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='Lagmann_data_path'):
        super(Lagemann, self).__init__(aug_params)
        self.split = split
        self.root = root
        self.is_test = (split == 'test')
        self.flow_types = ["VESSEL", "TWCF"]
        self.image_pairs = []
        self.missing_b_files = []
        self.sample_flow_types = []

        print("\n" + "="*100)
        print(f"Start loading {split} split dataset")
        print(f"Dataset root path: {root}")
        print(f"Flow types to match: {self.flow_types}")
        print("="*100)

        # Iterate over each flow type, check in detail
        for flow_type in self.flow_types:
            # Full path: root/flow_type/split (e.g., REAL_DATASET2/TCF/test)
            flow_split_dir = osp.join(root, flow_type, split)
            print(f"\n[Flow Type: {flow_type}]")
            print(f"Target directory: {flow_split_dir}")

            # 1. Check if directory exists
            if not osp.exists(flow_split_dir):
                print(f"❌ Directory does not exist! Skipping this flow type")
                continue

            # 2. Count all files in directory
            all_files = os.listdir(flow_split_dir)
            total_files = len(all_files)
            print(f"✅ Directory exists, total {total_files} files")

            # 3. Categorize by file extension (check for .jpg files)
            file_ext = {}
            for f in all_files:
                ext = osp.splitext(f)[1].lower()
                file_ext[ext] = file_ext.get(ext, 0) + 1
            print(f"File type distribution: {file_ext}")  # Output e.g., {'.jpg': 100, '.png': 20}

            # 4. Find files based on flow type matching rules
            if flow_type == "VESSEL":
                # Rule: .jpg files starting with C (e.g., C001.jpg, C002.jpg)
                match_files = sorted(glob(osp.join(flow_split_dir, "C*.jpg")))
                print(f"Matching rule: C*.jpg → Found {len(match_files)} files")
                # Generate image pairs (0&1, 2&3...)
                pairs = [(match_files[i], match_files[i+1]) for i in range(0, len(match_files)-1, 2)]

            elif flow_type == "TWCF":
                # Rule: *_A.jpg and corresponding *_B.jpg
                a_files = sorted(glob(osp.join(flow_split_dir, "*_A.jpg")))
                print(f"Matching rule: *_A.jpg → Found {len(a_files)} A-images")
                pairs = []
                missing_b = 0
                for a_f in a_files:
                    b_f = a_f.replace("_A.jpg", "_B.jpg")
                    if osp.exists(b_f):
                        pairs.append((a_f, b_f))
                    else:
                        missing_b += 1
                print(f"Valid A+B pairs: {len(pairs)} groups | Missing B-images: {missing_b}")

            elif flow_type == "TCF":
                # Rule: .jpg files starting with Dummy_M5_ (e.g., Dummy_M5_001.jpg)
                match_files = sorted(glob(osp.join(flow_split_dir, "Dummy_M5_*.jpg")))
                print(f"Matching rule: Dummy_M5_*.jpg → Found {len(match_files)} files")
                # Generate image pairs (0&1, 1&2...)
                pairs = [(match_files[i], match_files[i+1]) for i in range(len(match_files)-1)]

            else:
                pairs = []
                print(f"❌ No matching rule, skipping")

            # 5. Save image pairs
            self.image_pairs.extend(pairs)
            self.sample_flow_types.extend([flow_type]*len(pairs))
            print(f"Generated {len(pairs)} image pairs for this flow type")

        # Final statistics
        total_pairs = len(self.image_pairs)
        print(f"\n" + "="*100)
        print(f"{split} split loading complete → Total image pairs: {total_pairs} groups")
        print("="*100)

        if total_pairs == 0:
            # Raise clear error to guide user check
            raise ValueError(
                f"\n❌ No image pairs found! Please check the following:\n"
                f"1. Directory structure should be: {root}/[FlowType]/test (e.g., REAL_DATASET2/TCF/test)\n"
                f"2. Files must be in .jpg format (code only matches .jpg)\n"
                f"3. File naming must follow rules:\n"
                f"   - TCF: Dummy_M5_*.jpg (e.g., Dummy_M5_001.jpg)\n"
                f"   - VESSEL: C*.jpg (e.g., C001.jpg)\n"
                f"   - TWCF: *_A.jpg + *_B.jpg (e.g., img_001_A.jpg + img_001_B.jpg)\n"
                f"4. Are there enough images (at least 2) in each flow type's test directory?"
            )

        self.image_list = self.image_pairs
        self.flow_list = []
        self.extra_info = [[osp.basename(pair[0])] for pair in self.image_pairs]

    def __getitem__(self, index):
        # 1. Read image pair
            img1_path, img2_path = self.image_list[index]
            img1 = frame_utils.read_gen(img1_path)
            img2 = frame_utils.read_gen(img2_path)
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]

            # 2. Grayscale to RGB (if needed)
            if len(img1.shape) == 2:
                img1 = np.tile(img1[..., None], (1, 1, 3))
                img2 = np.tile(img2[..., None], (1, 1, 3))

            # 3. Convert to tensor
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            # 4. No GT flow: fill flow_gt with zeros, valid with ones
            flow_gt = torch.zeros(2, img1.shape[1], img1.shape[2], dtype=torch.float32)
            valid = torch.ones(img1.shape[1], img1.shape[2], dtype=torch.float32)

            # 5. Return 4 elements (Must!)
            return img1, img2, flow_gt, valid

class RealDataset(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='/home/dell/DATA/zhb/DATA/REAL_DATASET1/'):
        super(RealDataset, self).__init__(aug_params)
        self.split = split
        self.root = root
        self.flow_types = ['square', 'cylinder', 'backward']  # Define subfolders to process
        self.image_pairs = []  # Store all image pairs
        self.sample_subdirs = []  # Record source subdir for each sample (like Lagemann's sample_flow_types)
        self.invalid_subdirs = []  # Record invalid subdirs (insufficient or missing images)

        # Iterate through each subdir to process image pairs
        for subdir in self.flow_types:
            subdir_path = osp.join(root, subdir)

            # Check if subdir exists
            if not osp.exists(subdir_path):
                self.invalid_subdirs.append({
                    'subdir': subdir,
                    'reason': 'Folder does not exist'
                })
                continue

            # Get all images in folder
            images = glob(osp.join(subdir_path, 'Pic_*.tif'))
            if not images:
                self.invalid_subdirs.append({
                    'subdir': subdir,
                    'reason': 'No Pic_*.tif images found'
                })
                continue

            # Sort by numeric index in filename (Enhance robustness: ensure integer sorting)
            try:
                # Extract number after last underscore (e.g., "1" from "Pic_..._1.tif")
                images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            except (IndexError, ValueError) as e:
                self.invalid_subdirs.append({
                    'subdir': subdir,
                    'reason': f'Image sorting failed: {str(e)}'
                })
                continue

            # Check if image count is enough for at least 1 pair
            if len(images) < 2:
                self.invalid_subdirs.append({
                    'subdir': subdir,
                    'reason': f'Insufficient images (Total {len(images)}, need at least 2)'
                })
                continue

            # Generate consecutive image pairs (1&2, 2&3, ..., n-1&n)
            pairs = [(images[i], images[i + 1]) for i in range(len(images) - 1)]
            self.image_pairs.extend(pairs)
            self.sample_subdirs.extend([subdir] * len(pairs))  # Record source subdir for each sample

            # Output statistics for current subdir
            print(f"{subdir} {split} Statistics:")
            print(f"  Total Images: {len(images)}")
            print(f"  Generated Pairs: {len(pairs)}\n")

        # Output overall statistics
        total_pairs = len(self.image_pairs)
        print(f"All subdirs {split} generated total {total_pairs} image pairs")

        # Process invalid subdirs (Generate report)
        if self.invalid_subdirs:
            report_dir = osp.join(self.root, "dataset_report")
            os.makedirs(report_dir, exist_ok=True)
            report_path = osp.join(report_dir, f"RealDataset_{split}_invalid_subdirs.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"RealDataset {split} Invalid Subdirs Report (Total {len(self.invalid_subdirs)})\n")
                f.write("=" * 80 + "\n")
                for item in self.invalid_subdirs:
                    f.write(f"Subdir: {item['subdir']}\n")
                    f.write(f"Reason: {item['reason']}\n")
                    f.write("-" * 80 + "\n")
            print(f"Invalid subdirs report saved to: {report_path}")

        # Check for valid image pairs
        if total_pairs == 0:
            raise ValueError(f"No valid image pairs found in {split} split, check dataset structure and naming rules")

        # Initialize lists required by parent class
        self.image_list = self.image_pairs  # Image pair list (align with parent interface)
        self.flow_list = []  # No flow files, keep empty (refer to Lagemann processing)
        # Extra info: include source subdir and pair filenames (detailed for tracing samples)
        self.extra_info = [
            [subdir, osp.basename(pair[0]), osp.basename(pair[1])]
            for subdir, pair in zip(self.sample_subdirs, self.image_pairs)
        ]

    def __getitem__(self, index):
        if self.is_test:
            # Test mode: return image pair and extra info only
            img1_path, img2_path = self.image_list[index]
            img1 = frame_utils.read_gen(img1_path)
            img2 = frame_utils.read_gen(img2_path)

            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        # Training mode: init random seed (same logic as parent)
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        img1_path, img2_path = self.image_list[index]

        # Read and process images
        img1 = frame_utils.read_gen(img1_path)
        img2 = frame_utils.read_gen(img2_path)

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # Grayscale to RGB (same logic as parent)
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # Apply data augmentation (compatible with augmentor interface)
        if self.augmentor is not None:
            try:
                # Assume augmentor accepts (img1, img2) input (no flow)
                img1, img2 = self.augmentor(img1, img2)
            except Exception as e:
                print(f"Data augmentation failed (Sample {self.extra_info[index]}): {e}, skipping augmentation")

        # Convert to PyTorch tensor
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        # No flow data: generate zero flow tensor and full valid mask (consistent with parent return format)
        flow = torch.zeros(2, img1.shape[1], img1.shape[2], dtype=torch.float32)
        valid = torch.ones(img1.shape[1], img1.shape[2], dtype=torch.float32)

        return img1, img2, flow, valid
# ----------------------
# Step 2: Data Loading Function (Adapted for Lagemann Dataset)
# ----------------------
def fetch_dataloader(args):
    aug_params = None
    val_dataset = Lagemann(
        aug_params=aug_params,
        split='test',
        root=args.test_data_root
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    print(f"Test set samples: {len(val_dataset)} image pairs")
    return val_loader  # Return only 1 value

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False
try:
    font_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            from matplotlib.font_manager import FontProperties

            plt.rcParams["font.family"] = FontProperties(fname=font_path).get_name()
            break
except:
    pass


def parula_cmap():
    colors = [
        [0.2422, 0.1504, 0.6603], [0.2501, 0.2501, 0.7500], [0.2023, 0.3809, 0.8100],
        [0.1464, 0.4956, 0.8100], [0.0952, 0.5944, 0.7900], [0.0594, 0.6770, 0.7533],
        [0.0462, 0.7455, 0.7067], [0.0581, 0.8019, 0.6500], [0.1025, 0.8475, 0.5867],
        [0.1773, 0.8822, 0.5150], [0.2809, 0.9084, 0.4350], [0.4096, 0.9268, 0.3400],
        [0.5611, 0.9370, 0.2275], [0.7201, 0.9399, 0.1312], [0.8747, 0.9360, 0.0533],
        [0.9966, 0.9303, 0.0031]
    ]
    return LinearSegmentedColormap.from_list('parula', colors, N=256)


MAX_FLOW = 400


def calculate_sample_metrics(flow_pred, flow_gt, valid):
    flow_gt_mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
    valid_mask = (valid >= 0.5) & (flow_gt_mag < MAX_FLOW)
    if valid_mask.sum() == 0:
        return 0.0, 0.0, 0.0

    flow_diff = flow_pred - flow_gt
    epe_per_pixel = torch.sqrt(torch.sum(flow_diff ** 2, dim=0))
    aee = round(epe_per_pixel[valid_mask].mean().item(), 6)

    mae_x = torch.abs(flow_diff[0, :, :][valid_mask]).mean().item()
    mae_y = torch.abs(flow_diff[1, :, :][valid_mask]).mean().item()
    mae = round((mae_x + mae_y) / 2, 6)

    mse_x = (flow_diff[0, :, :][valid_mask] ** 2).mean().item()
    mse_y = (flow_diff[1, :, :][valid_mask] ** 2).mean().item()
    mse = round((mse_x + mse_y) / 2, 6)

    return aee, mae, mse


def visualize_flow_heatmap(sample_idx, flow_pred, flow_gt, valid, save_dir, aee, mae, mse,
                           max_mag, arrow_step=10, cmap='jet', flow_type=None, show_arrows=True):
    flow_pred_np = flow_pred.permute(1, 2, 0).cpu().numpy()
    flow_gt_np = flow_gt.permute(1, 2, 0).cpu().numpy()
    valid_np = valid.cpu().numpy()
    height, width = valid_np.shape

    pred_mag = np.sqrt(flow_pred_np[..., 0] ** 2 + flow_pred_np[..., 1] ** 2)
    gt_mag = np.sqrt(flow_gt_np[..., 0] ** 2 + flow_gt_np[..., 1] ** 2)
    valid_mask = (valid_np >= 0.5) & (gt_mag < MAX_FLOW)
    pred_mag[~valid_mask] = np.nan
    gt_mag[~valid_mask] = np.nan

    pred_mag_norm = np.clip(pred_mag / max_mag, 0, 1) if max_mag != 0 else pred_mag
    gt_mag_norm = np.clip(gt_mag / max_mag, 0, 1) if max_mag != 0 else gt_mag

    cmap_obj = parula_cmap() if cmap == 'parula' else plt.cm.jet
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(pred_mag_norm, cmap=cmap_obj, origin='upper')
    ax1.set_title(f'{flow_type} - Sample {sample_idx} - Predicted Flow\nAEE: {aee:.6f}', fontsize=12)

    # Draw arrows when show_arrows is True
    if show_arrows:
        x = np.arange(0, width, arrow_step)
        y = np.arange(0, height, arrow_step)
        xx, yy = np.meshgrid(x, y)
        u_pred = flow_pred_np[yy, xx, 0].copy()
        v_pred = flow_pred_np[yy, xx, 1].copy()
        u_pred[~valid_mask[yy, xx]] = 0.0
        v_pred[~valid_mask[yy, xx]] = 0.0
        pred_mag_arrow = np.sqrt(u_pred ** 2 + v_pred ** 2)
        if np.max(pred_mag_arrow) > 0:
            scale_factor = (arrow_step / 2) / np.max(pred_mag_arrow)
            u_pred_scaled = u_pred * scale_factor
            v_pred_scaled = v_pred * scale_factor
        else:
            u_pred_scaled = u_pred
            v_pred_scaled = v_pred
        ax1.quiver(xx, yy, u_pred_scaled, v_pred_scaled, color='k',
                   scale=0.5, scale_units='xy', angles='xy', width=0.0015)
    ax1.axis('off')

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(f'Speed Magnitude (Normalized)', fontsize=10)

    im2 = ax2.imshow(gt_mag_norm, cmap=cmap_obj, origin='upper')
    ax2.set_title(f'{flow_type} - Sample {sample_idx} - GT Flow\nMax Magnitude: {max_mag:.2f}', fontsize=12)

    # Draw arrows when show_arrows is True
    if show_arrows:
        x = np.arange(0, width, arrow_step)
        y = np.arange(0, height, arrow_step)
        xx, yy = np.meshgrid(x, y)
        u_gt = flow_gt_np[yy, xx, 0].copy()
        v_gt = flow_gt_np[yy, xx, 1].copy()
        u_gt[~valid_mask[yy, xx]] = 0.0
        v_gt[~valid_mask[yy, xx]] = 0.0
        gt_mag_arrow = np.sqrt(u_gt ** 2 + v_gt ** 2)
        if np.max(gt_mag_arrow) > 0:
            scale_factor_gt = (arrow_step / 2) / np.max(gt_mag_arrow)
            u_gt_scaled = u_gt * scale_factor_gt
            v_gt_scaled = v_gt * scale_factor_gt
        else:
            u_gt_scaled = u_gt
            v_gt_scaled = v_gt
        ax2.quiver(xx, yy, u_gt_scaled, v_gt_scaled, color='k',
                   scale=0.5, scale_units='xy', angles='xy', width=0.0015)
    ax2.axis('off')

    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label(f'Speed Magnitude (Normalized)', fontsize=10)

    flow_heatmap_dir = os.path.join(save_dir, flow_type)
    os.makedirs(flow_heatmap_dir, exist_ok=True)
    save_path = os.path.join(flow_heatmap_dir, f'sample_{sample_idx}_heatmap.png')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_flow_on_axis(ax, flow, valid, speed_threshold=0.0,
                           streamline_density=[1.5, 1.5], streamline_linewidth=2.0,
                           streamline_minlength=0.2, streamline_maxlength=1.5, cmap='jet'):
    H, W, _ = flow.shape
    flow_magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    valid_mask = (valid >= 0.5) & (flow_magnitude < MAX_FLOW)
    flow_magnitude[~valid_mask] = np.nan

    U = flow[..., 0].copy()
    V = flow[..., 1].copy()
    low_speed_mask = (flow_magnitude <= speed_threshold) | np.isnan(flow_magnitude)
    U[low_speed_mask] = 0.0
    V[low_speed_mask] = 0.0
    masked_magnitude = flow_magnitude.copy()
    masked_magnitude[low_speed_mask] = np.nan

    valid_mag = masked_magnitude[~np.isnan(masked_magnitude)]
    max_mag = np.percentile(valid_mag, 95) if len(valid_mag) > 0 else 1.0
    mag_norm = np.clip(masked_magnitude / max_mag, 0, 1)

    cmap_obj = parula_cmap() if cmap == 'parula' else plt.cm.jet
    X = np.arange(W)
    Y = np.arange(H)
    strm = ax.streamplot(
        X, Y, U, V,
        density=streamline_density,
        color=mag_norm,
        cmap=cmap_obj,
        linewidth=streamline_linewidth,
        arrowsize=0.5,
        minlength=streamline_minlength,
        maxlength=streamline_maxlength,
        norm=plt.Normalize(vmin=0, vmax=1)
    )

    ax.axis('off')
    return ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap=cmap_obj)


def visualize_flow_streamline(sample_idx, img1, img2, flow_pred, flow_gt, valid, save_dir,
                              aee, mae, mse, streamline_params, max_mag, flow_type=None):
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    img1_np = (img1_np * 255).clip(0, 255).astype(np.uint8)
    img2_np = (img2_np * 255).clip(0, 255).astype(np.uint8)

    flow_pred_np = flow_pred.permute(1, 2, 0).cpu().numpy()
    flow_gt_np = flow_gt.permute(1, 2, 0).cpu().numpy()
    valid_np = valid.cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    cmap = streamline_params.get('cmap', 'jet')

    axes[0].imshow(img1_np, origin='lower')
    axes[0].set_title(f"{flow_type} - Frame 1", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(img2_np, origin='lower')
    axes[1].set_title(f"{flow_type} - Frame 2", fontsize=12)
    axes[1].axis('off')

    pred_sm = visualize_flow_on_axis(axes[2], flow_pred_np, valid_np, **streamline_params)
    axes[2].set_title(f"{flow_type} - Predicted Streamline\nAEE: {aee:.6f}", fontsize=12)
    fig.colorbar(pred_sm, ax=axes[2], fraction=0.046, pad=0.04, label=f'Flow Velocity (Normalized)')

    gt_sm = visualize_flow_on_axis(axes[3], flow_gt_np, valid_np, **streamline_params)
    axes[3].set_title(f"{flow_type} - GT Streamline\nMax Magnitude: {max_mag:.2f}", fontsize=12)
    fig.colorbar(gt_sm, ax=axes[3], fraction=0.046, pad=0.04, label=f'Flow Velocity (Normalized)')

    param_text = f"Density: {streamline_params['streamline_density'][0]}x{streamline_params['streamline_density'][1]} | " \
                 f"Width: {streamline_params['streamline_linewidth']} | Threshold: {streamline_params['speed_threshold']}"
    fig.suptitle(f"{flow_type} - Sample {sample_idx} Flow Streamline Comparison | {param_text}", fontsize=14, y=0.98)

    flow_streamline_dir = os.path.join(save_dir, flow_type)
    os.makedirs(flow_streamline_dir, exist_ok=True)
    save_path = os.path.join(flow_streamline_dir, f'sample_{sample_idx}_streamline.png')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_flow_metrics_curve(flow_metrics_dict, save_dir):
    flow_types = list(flow_metrics_dict.keys())
    avg_aee_list = [flow_metrics_dict[ft]['avg_aee'] for ft in flow_types]
    avg_mae_list = [flow_metrics_dict[ft]['avg_mae'] for ft in flow_types]
    avg_mse_list = [flow_metrics_dict[ft]['avg_mse'] for ft in flow_types]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(len(flow_types))
    width = 0.25

    ax.bar(x - width, avg_aee_list, width, label='Avg AEE', color='#1f77b4')
    ax.bar(x, avg_mae_list, width, label='Avg MAE', color='#ff7f0e')
    ax.bar(x + width, avg_mse_list, width, label='Avg MSE', color='#2ca02c')

    ax.set_xticks(x)
    ax.set_xticklabels(flow_types, rotation=45, ha='right')
    ax.set_xlabel('Flow Type', fontsize=12)
    ax.set_ylabel('Average Metric Value (Lower is Better)', fontsize=12)
    ax.set_title('Comparison of Average Flow Metrics Across Flow Types', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    ax.ticklabel_format(axis='y', useOffset=False, style='plain')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'flow_metrics_comparison.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Flow metrics comparison plot saved to: {save_path}")

def test(args):
    # 1. Create result directories
    result_root = os.path.join(args.output_dir, "test_results")
    metrics_dir = os.path.join(result_root, "metrics")
    flow_metrics_dir = os.path.join(metrics_dir, "flow_metrics")
    heatmap_dir = os.path.join(result_root, "heatmaps")
    streamline_dir = os.path.join(result_root, "streamlines")
    curve_dir = os.path.join(result_root, "curves")
    for dir_path in [metrics_dir, flow_metrics_dir, heatmap_dir, streamline_dir, curve_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 2. Organize parameters
    streamline_params = {
        'streamline_density': args.streamline_density,
        'streamline_linewidth': args.streamline_linewidth,
        'speed_threshold': args.speed_threshold,
        'streamline_minlength': args.streamline_minlength,
        'streamline_maxlength': args.streamline_maxlength,
        'cmap': args.cmap
    }

    # 3. Initialize model
    model = nn.DataParallel(AFKAN_PIV(args), device_ids=args.gpus)
    if not os.path.exists(args.restore_ckpt):
        raise ValueError(f"Model checkpoint does not exist! Path: {args.restore_ckpt}")
    print(f"Loading model checkpoint: {args.restore_ckpt}")
    model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
    model.cuda()
    model.eval()

    # 4. Load test dataset
    print(f"Loading test dataset: {args.test_data_root}")
    test_loader = fetch_dataloader(args)
    test_dataset = test_loader.dataset
    total_samples = len(test_dataset)
    print(f"Total samples in test set: {total_samples}")

    # 5. Parse flow types
    sample_img1_paths = [item[0] for item in test_dataset.image_list]
    if not sample_img1_paths:
        raise ValueError("Test dataset image_list is empty, no sample paths loaded!")

    flow_types = test_dataset.flow_types
    print(f"Detected flow types in test set: {flow_types} (Total {len(flow_types)})")

    # Critical optimization: Sort flow type names by length descending, prioritize longer names for matching
    sorted_flow_types = sorted(flow_types, key=lambda x: len(x), reverse=True)

    flow_sample_idx_map = {ft: [] for ft in flow_types}
    unmatched_samples = []
    for sample_idx, img1_path in enumerate(sample_img1_paths):
        matched = False
        # Use sorted flow types list for matching
        for flow_type in sorted_flow_types:
            if flow_type in img1_path:
                flow_sample_idx_map[flow_type].append(sample_idx)
                matched = True
                break
        if not matched:
            unmatched_samples.append(img1_path)

    if unmatched_samples:
        print(f"Warning: {len(unmatched_samples)} samples did not match any flow type, example paths:")
        for path in unmatched_samples[:3]:
            print(f"  {path}")

    valid_flow_types = [ft for ft in flow_types if len(flow_sample_idx_map[ft]) > 0]
    if not valid_flow_types:
        raise ValueError("No samples matched any flow type, please check directory structure!")
    print(f"Sample count per flow type: { {ft: len(idx_list) for ft, idx_list in flow_sample_idx_map.items()} }")

    # 6. Initialize flow metrics dictionary
    flow_metrics_dict = {
        ft: {
            'total_samples': len(flow_sample_idx_map[ft]),
            'valid_samples': 0,
            'aee_list': [],
            'mae_list': [],
            'mse_list': [],
            'avg_aee': 0.0,
            'avg_mae': 0.0,
            'avg_mse': 0.0
        } for ft in flow_types
    }

    # 7. Process samples by flow type (Enable block processing for TWCF)
    with torch.no_grad():
        total_flow_count = len(valid_flow_types)
        for flow_idx, flow_type in enumerate(valid_flow_types):
            sample_idx_list = flow_sample_idx_map[flow_type]
            flow_sample_count = len(sample_idx_list)
            print(f"\n=== Processing Flow Type [{flow_idx + 1}/{total_flow_count}]: {flow_type} (Total {flow_sample_count} samples) ===")

            # Enable block processing for TWCF, normal processing for other flow types
            use_block_process = (flow_type == "TWCF")
            if use_block_process:
                print(f"Block processing enabled: Block size {args.block_size}, Overlap {args.overlap} pixels")

            with tqdm.tqdm(total=flow_sample_count, desc=f"{flow_type} Sample Processing") as pbar:
                for idx_in_flow, sample_idx in enumerate(sample_idx_list):
                    # Load sample
                    data = test_dataset[sample_idx]
                    img1, img2, flow_gt, valid = data
                    H, W = img1.shape[1], img1.shape[2]  # Original image size [C, H, W]
                    device = img1.device if hasattr(img1, 'device') else 'cuda'

                    # --------------------
                    # Block Processing Logic (TWCF Only)
                    # --------------------
                    if use_block_process:
                        # 1. Split image into blocks
                        img1_blocks, coords = split_image_into_blocks(
                            img1, block_size=args.block_size, overlap=args.overlap
                        )
                        img2_blocks, _ = split_image_into_blocks(
                            img2, block_size=args.block_size, overlap=args.overlap
                        )

                        # 2. Predict flow for each block
                        flow_pred_blocks = []
                        for b1, b2 in zip(img1_blocks, img2_blocks):
                            # Add batch dimension and move to GPU
                            b1_cuda = b1.unsqueeze(0).cuda()
                            b2_cuda = b2.unsqueeze(0).cuda()
                            # Model prediction
                            b_preds = model(b1_cuda, b2_cuda, iters=args.iters)
                            b_flow = b_preds[-1].squeeze(0)  # [2, block_h, block_w]
                            flow_pred_blocks.append(b_flow)

                        # 3. Merge block flows into full flow field
                        flow_pred = merge_flow_blocks(
                            flow_pred_blocks,
                            coords,
                            original_shape=(H, W),
                            overlap=args.overlap,
                            device=torch.device('cuda')
                        )

                        # 4. Move GT and valid to GPU (Keep consistent with predicted flow field)
                        flow_gt = flow_gt.cuda()
                        valid = valid.cuda()

                    # --------------------
                    # Normal Processing Logic (Non-TWCF)
                    # --------------------
                    else:
                        img1_cuda = img1.unsqueeze(0).cuda()
                        img2_cuda = img2.unsqueeze(0).cuda()
                        flow_gt = flow_gt.cuda()
                        valid = valid.cuda()
                        # Direct prediction
                        flow_preds = model(img1_cuda, img2_cuda, iters=args.iters)
                        flow_pred = flow_preds[-1].squeeze(0)

                    # Calculate metrics
                    aee, mae, mse = calculate_sample_metrics(flow_pred, flow_gt, valid)

                    # Calculate max magnitude
                    flow_gt_np = flow_gt.permute(1, 2, 0).cpu().numpy()
                    gt_mag = np.sqrt(flow_gt_np[..., 0] ** 2 + flow_gt_np[..., 1] ** 2)
                    valid_np = valid.cpu().numpy()
                    valid_mask = (valid_np >= 0.5) & (gt_mag < MAX_FLOW)
                    valid_pixel_count = valid_mask.sum()
                    valid_gt_mag = gt_mag[valid_mask]
                    max_mag = np.percentile(valid_gt_mag, 95) if len(valid_gt_mag) > 0 else 0.0

                    # Update metric statistics
                    if valid_pixel_count > 0:
                        flow_metrics_dict[flow_type]['valid_samples'] += 1
                        flow_metrics_dict[flow_type]['aee_list'].append(aee)
                        flow_metrics_dict[flow_type]['mae_list'].append(mae)
                        flow_metrics_dict[flow_type]['mse_list'].append(mse)

                    # Generate visualization (Pass arrow switch parameter)
                    visualize_flow_heatmap(
                        sample_idx=idx_in_flow,
                        flow_pred=flow_pred,
                        flow_gt=flow_gt,
                        valid=valid,
                        save_dir=heatmap_dir,
                        aee=aee,
                        mae=mae,
                        mse=mse,
                        max_mag=max_mag,
                        arrow_step=args.arrow_step,
                        cmap=args.cmap,
                        flow_type=flow_type,
                        show_arrows=args.show_arrows  # Control arrow display (True/False)
                    )

                    visualize_flow_streamline(
                        sample_idx=idx_in_flow,
                        img1=img1,
                        img2=img2,
                        flow_pred=flow_pred,
                        flow_gt=flow_gt,
                        valid=valid,
                        save_dir=streamline_dir,
                        aee=aee,
                        mae=mae,
                        mse=mse,
                        streamline_params=streamline_params,
                        max_mag=max_mag,
                        flow_type=flow_type
                    )

                    # Update progress bar
                    pbar.set_postfix({
                        "FlowType": flow_type,
                        "Sample": f"{idx_in_flow + 1}/{flow_sample_count}",
                        "AEE": f"{aee:.6f}",
                        "ValidSamples": flow_metrics_dict[flow_type]['valid_samples']
                    })
                    pbar.update(1)

            # Calculate average metrics for flow type
            valid_count = flow_metrics_dict[flow_type]['valid_samples']
            if valid_count > 0:
                flow_metrics_dict[flow_type]['avg_aee'] = round(np.mean(flow_metrics_dict[flow_type]['aee_list']), 6)
                flow_metrics_dict[flow_type]['avg_mae'] = round(np.mean(flow_metrics_dict[flow_type]['mae_list']), 6)
                flow_metrics_dict[flow_type]['avg_mse'] = round(np.mean(flow_metrics_dict[flow_type]['mse_list']), 6)
            print(f"=== Flow Type {flow_type} Processing Complete ===")
            print(f"Total Samples: {flow_sample_count} | Valid Samples: {valid_count}")
            print(f"Average AEE: {flow_metrics_dict[flow_type]['avg_aee']:.6f}")
            print(f"Average MAE: {flow_metrics_dict[flow_type]['avg_mae']:.6f}")
            print(f"Average MSE: {flow_metrics_dict[flow_type]['avg_mse']:.6f}\n")

    # 8. Save metrics
    flow_metrics_summary = []
    for flow_type, metrics in flow_metrics_dict.items():
        if metrics['total_samples'] == 0:
            continue
        flow_metric_path = os.path.join(flow_metrics_dir, f"{flow_type}_metrics.txt")
        with open(flow_metric_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Flow Type: {flow_type} Detailed Metrics\n")
            f.write("=" * 80 + "\n")
            f.write(f"Test Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Checkpoint Path: {args.restore_ckpt}\n")
            f.write(f"Total Samples: {metrics['total_samples']}\n")
            f.write(f"Valid Samples: {metrics['valid_samples']}/{metrics['total_samples']}\n")
            if flow_type == "TWCF":
                f.write(f"Block Params: Block Size {args.block_size} | Overlap {args.overlap} pixels\n")
            f.write(f"Arrow Display: {'Enabled' if args.show_arrows else 'Disabled'}\n")  # Record arrow switch status
            f.write("=" * 80 + "\n")
            f.write(f"Average AEE: {metrics['avg_aee']:.6f}\n")
            f.write(f"Average MAE: {metrics['avg_mae']:.6f}\n")
            f.write(f"Average MSE: {metrics['avg_mse']:.6f}\n")
            f.write(f"AEE Std Dev: {round(np.std(metrics['aee_list']), 6) if metrics['valid_samples'] > 0 else 0.0:.6f}\n")
            f.write(f"MAE Std Dev: {round(np.std(metrics['mae_list']), 6) if metrics['valid_samples'] > 0 else 0.0:.6f}\n")
            f.write("=" * 80 + "\n")
        print(f"Flow Type {flow_type} Metrics file saved to: {flow_metric_path}")
        flow_metrics_summary.append({
            'FlowTypeName': flow_type,
            'TotalSamples': metrics['total_samples'],
            'ValidSamples': metrics['valid_samples'],
            'AvgAEE': metrics['avg_aee'],
            'AvgMAE': metrics['avg_mae'],
            'AvgMSE': metrics['avg_mse'],
            'AEE_StdDev': round(np.std(metrics['aee_list']), 6) if metrics['valid_samples'] > 0 else 0.0
        })

    # Summary CSV
    if flow_metrics_summary:
        flow_summary_df = pd.DataFrame(flow_metrics_summary)
        flow_summary_path = os.path.join(metrics_dir, "flow_metrics_summary.csv")
        flow_summary_df.to_csv(flow_summary_path, index=False, encoding='utf-8-sig')
        print(f"\nFlow metrics summary CSV saved to: {flow_summary_path}")

    # 9. Global Metrics
    global_total_samples = sum([m['total_samples'] for m in flow_metrics_dict.values()])
    global_valid_samples = sum([m['valid_samples'] for m in flow_metrics_dict.values()])
    global_aee_list = sum([m['aee_list'] for m in flow_metrics_dict.values()], [])
    global_avg_aee = round(np.mean(global_aee_list), 6) if global_valid_samples > 0 else 0.0
    global_avg_mae = round(np.mean(sum([m['mae_list'] for m in flow_metrics_dict.values()], [])),
                           6) if global_valid_samples > 0 else 0.0
    global_avg_mse = round(np.mean(sum([m['mse_list'] for m in flow_metrics_dict.values()], [])),
                           6) if global_valid_samples > 0 else 0.0

    # Save Global Metrics
    global_metric_path = os.path.join(metrics_dir, "global_metrics.txt")
    with open(global_metric_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("AFKAN Model Test Set Global Metrics\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Checkpoint Path: {args.restore_ckpt}\n")
        f.write(f"Number of Flow Types: {len(valid_flow_types)}\n")
        f.write(f"Global Total Samples: {global_total_samples}\n")
        f.write(f"Global Valid Samples: {global_valid_samples}/{global_total_samples}\n")
        f.write(f"Arrow Display: {'Enabled' if args.show_arrows else 'Disabled'}\n")  # Record arrow switch status
        f.write("=" * 80 + "\n")
        f.write(f"Global Average AEE: {global_avg_aee:.6f}\n")
        f.write(f"Global Average MAE: {global_avg_mae:.6f}\n")
        f.write(f"Global Average MSE: {global_avg_mse:.6f}\n")
        f.write("=" * 80 + "\n")
    print(f"Global Metrics file saved to: {global_metric_path}")

    # 10. Plot Metrics Comparison
    if valid_flow_types:
        plot_flow_metrics_curve(flow_metrics_dict, curve_dir)

    # 11. Print Final Results
    print("\n" + "=" * 80)
    print("AFKAN Model Test Complete!")
    print(f"Number of Flow Types: {len(valid_flow_types)} | Global Total Samples: {global_total_samples}")
    print(f"Global Avg AEE: {global_avg_aee:.6f} | Global Avg MAE: {global_avg_mae:.6f} | Global Avg MSE: {global_avg_mse:.6f}")
    print(f"Heatmap Arrow Display: {'Enabled' if args.show_arrows else 'Disabled'}")  # Display arrow switch status
    print(f"Result Root Directory: {result_root}")
    print("=" * 80)


# ----------------------
# Step 5: Command Line Parameters (Modified to True/False for arrow control)
# ----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AFKAN Testing Script")

    # Model and Checkpoint
    parser.add_argument('--restore_ckpt', required=True,
                        help="Path to pre-trained model checkpoint (Required).")
    parser.add_argument('--small', action='store_true',
                        help="Whether to use the small model variant.")
    parser.add_argument('--iters', type=int, default=1,
                        help="Number of model iterations.")

    # Model Architecture
    parser.add_argument('--dim', type=int, default=128,
                        help="Feature dimension size.")
    parser.add_argument('--radius', type=int, default=2,
                        help="Attention radius.")
    parser.add_argument('--num_blocks', type=int, default=3,
                        help="Number of network blocks.")
    parser.add_argument('--block_dims', type=int, nargs='+', default=[64, 128, 256],
                        help="Dimensions for each block.")
    parser.add_argument('--initial_dim', type=int, default=64,
                        help="Initial feature dimension.")
    parser.add_argument('--pretrain', default='resnet34',
                        help="Backbone network (resnet34/resnet18).")

    # Data Parameters
    parser.add_argument('--stage', default='simdata',
                        help="Dataset type (for compatibility).")
    parser.add_argument('--data_root', type=str,
                        default='data_path_maybe_nothing',
                        help="Data root path (for compatibility).")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 256],
                        help="Input image size (H, W).")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Test batch size (Recommend 1 for block processing).")
    parser.add_argument('--use_var', type=bool, default=False,
                        help="Manually define displacement range.")
    parser.add_argument('--var_max', type=int, default=20,
                        help="Max displacement (active if use_var=True).")
    parser.add_argument('--var_min', type=int, default=0,
                        help="Min displacement (active if use_var=True).")
    parser.add_argument('--do_flip', action='store_true',
                        help="Data flipping (Recommend disabled for testing).")

    # Block Processing Parameters (TWCF only)
    parser.add_argument('--block_size', type=int, nargs='+', default=[256, 256],
                        help="Block size (H, W), recommend 512x512 or 256x256.")
    parser.add_argument('--overlap', type=int, default=64,
                        help="Overlap pixels between blocks (Recommend 64 or 128).")

    # Output Parameters
    parser.add_argument('--output_dir', type=str,
                        default='output_path',
                        help="Root directory to save test results.")
    parser.add_argument('--test_data_root', type=str,
                        default='test_path',
                        help='Root directory for Lagemann dataset.')

    # Visualization Parameters
    parser.add_argument('--arrow_step', type=int, default=10,
                        help="Arrow interval step for heatmap.")
    parser.add_argument('--cmap', type=str, default='jet', choices=['jet', 'parula'],
                        help="Colormap for visualization.")
    parser.add_argument('--streamline_density', type=float, nargs='+', default=[2, 2],
                        help="Streamline density.")
    parser.add_argument('--streamline_linewidth', type=float, default=1,
                        help="Streamline line width.")
    parser.add_argument('--speed_threshold', type=float, default=0.01,
                        help="Speed threshold for streamlines.")
    parser.add_argument('--streamline_minlength', type=float, default=0.1,
                        help="Minimum streamline length.")
    parser.add_argument('--streamline_maxlength', type=float, default=6,
                        help="Maximum streamline length.")
    # Arrow Switch: True to enable, False to disable (Default True)
    parser.add_argument('--show_arrows', type=bool, default=True,
                        help="Show arrows in heatmap (True=Show, False=Hide, Default=True).")

    # Device Parameters
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                        help="GPU IDs to use.")

    args = parser.parse_args()

    # Fix random seed
    torch.manual_seed(1234)
    np.random.seed(1234)

    # Execute test
    test(args)