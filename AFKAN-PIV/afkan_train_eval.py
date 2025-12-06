from __future__ import print_function, division
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model.AFKAN_PIV.afkan_piv import AFKAN_PIV
from core.datasets import fetch_dataloader, Wallshear  # Import dataset classes
from torch.utils.tensorboard import SummaryWriter
import tqdm
import argparse
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # Added: For custom color mapping
from PIL import Image
import cv2

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

MAX_FLOW = 400
SUM_FREQ = 1000
VAL_FREQ = 5000
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ----------------------
# Added: Custom MATLAB-style parula colormap (Optional)
# ----------------------
def parula_cmap():
    """Simulate MATLAB's parula color map"""
    colors = [
        [0.2422, 0.1504, 0.6603], [0.2501, 0.2501, 0.7500], [0.2023, 0.3809, 0.8100],
        [0.1464, 0.4956, 0.8100], [0.0952, 0.5944, 0.7900], [0.0594, 0.6770, 0.7533],
        [0.0462, 0.7455, 0.7067], [0.0581, 0.8019, 0.6500], [0.1025, 0.8475, 0.5867],
        [0.1773, 0.8822, 0.5150], [0.2809, 0.9084, 0.4350], [0.4096, 0.9268, 0.3400],
        [0.5611, 0.9370, 0.2275], [0.7201, 0.9399, 0.1312], [0.8747, 0.9360, 0.0533],
        [0.9966, 0.9303, 0.0031]
    ]
    return LinearSegmentedColormap.from_list('parula', colors, N=256)


# ----------------------
# Loss functions and evaluation metrics
# ----------------------
def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return flow_loss, metrics


# ----------------------
# Calculate loss between prediction and GT
# ----------------------
def calculate_gt_losses(pred_flow, gt_flow):
    if isinstance(pred_flow, np.ndarray):
        pred_flow = torch.from_numpy(pred_flow).float()
    if isinstance(gt_flow, np.ndarray):
        gt_flow = torch.from_numpy(gt_flow).float()

    epe = torch.sqrt(torch.sum((pred_flow - gt_flow) ** 2, dim=-1)).mean().item()
    mse = torch.mean((pred_flow - gt_flow) ** 2).item()
    diff = torch.sqrt(torch.sum((pred_flow - gt_flow) ** 2, dim=-1))
    px1 = (diff < 1).float().mean().item()
    px3 = (diff < 3).float().mean().item()
    px5 = (diff < 5).float().mean().item()

    return {
        'epe': epe, 'mse': mse, '1px': px1, '3px': px3, '5px': px5
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    total_steps = args.epochs * len(fetch_dataloader(args)[0]) + 100
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, total_steps, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear'
    )
    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler, log_type):
        self.model = model
        self.scheduler = scheduler
        self.total_epochs = 0
        self.writer = None
        self.log_type = log_type  # Distinguish between train/test logs

    def write_epoch_dict(self, results):
        if self.writer is None:
            current_time = datetime.datetime.now().strftime(f"AFKAN"
                                                            f"{self.log_type}_%Y%m%d-%H%M%S")
            log_dir = os.path.join(args.output_dir, 'log', current_time)
            self.writer = SummaryWriter(log_dir)
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_epochs)
        self.total_epochs += 1

    def close(self):
        if self.writer:
            self.writer.close()


def visualize_flow_on_axis(ax, flow, arrow_step=10, title=None, cmap='jet'):
    """Draw heatmap-style optical flow on the given matplotlib axis (color determined by speed magnitude only)"""
    height, width, _ = flow.shape

    # Calculate flow magnitude (sole factor for color)
    flow_magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # Normalize speed magnitude to [0, 1] range (for color mapping)
    max_mag = np.max(flow_magnitude)
    if max_mag == 0:
        flow_magnitude_normalized = np.zeros_like(flow_magnitude)
    else:
        flow_magnitude_normalized = flow_magnitude / max_mag

    # Select color map (supports 'jet' or 'parula')
    if cmap == 'parula':
        cmap_obj = parula_cmap()
    else:  # Default to jet
        cmap_obj = plt.cm.jet

    # Draw heatmap (origin at top-left, matching MATLAB)
    im = ax.imshow(flow_magnitude_normalized, cmap=cmap_obj, origin='upper')

    # Add arrows (control density)
    x = np.arange(0, width, arrow_step)
    y = np.arange(0, height, arrow_step)
    xx, yy = np.meshgrid(x, y)

    # Extract flow values at arrow positions
    u = flow[yy, xx, 0]
    v = flow[yy, xx, 1]
    magnitudes = np.sqrt(u ** 2 + v ** 2)

    # Normalize arrow length
    if np.max(magnitudes) > 0:
        scale_factor = (arrow_step / 2) / np.max(magnitudes)  # Max arrow length is approx half the step
        u_scaled = u * scale_factor
        v_scaled = v * scale_factor
    else:
        u_scaled = u
        v_scaled = v

    # Draw black arrows (clearer contrast with heatmap)
    ax.quiver(xx, yy, u_scaled, v_scaled, color='k',
              scale=0.5, scale_units='xy', angles='xy', width=0.0015)

    # Set title and axes
    if title:
        ax.set_title(title)
    ax.axis('off')
    return im  # Return image object for adding colorbar


def visualize_flow_comparison(img1, img2, pred_flow, gt_flow=None, save_path=None, title="",
                              arrow_step=10, losses=None, cmap='jet'):
    """Visualize image pairs, predicted flow, and GT flow (heatmap style)"""
    # Convert tensor to numpy array (H, W, C)
    if isinstance(img1, torch.Tensor):
        img1 = img1.permute(1, 2, 0).cpu().numpy()  # (C,H,W) -> (H,W,C)
        img2 = img2.permute(1, 2, 0).cpu().numpy()
        pred_flow = pred_flow.permute(1, 2, 0).cpu().numpy()
        if gt_flow is not None:
            gt_flow = gt_flow.permute(1, 2, 0).cpu().numpy()

    # Normalize images (ensure range [0, 255])
    img1 = (img1 * 255).clip(0, 255).astype(np.uint8)
    img2 = (img2 * 255).clip(0, 255).astype(np.uint8)

    # Adjust number of subplots (whether to include GT flow)
    has_gt = gt_flow is not None
    cols = 4 if has_gt else 3
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))

    # Build title (including loss information)
    if losses and has_gt:
        loss_text = f"EPE: {losses['epe']:.4f} | MSE: {losses['mse']:.4f} | "
        loss_text += f"1px: {losses['1px']:.2%} | 3px: {losses['3px']:.2%}"
        full_title = f"{title}\n{loss_text}"
    else:
        full_title = title

    if full_title:
        fig.suptitle(full_title, fontsize=12)

    # Display Frame 1
    axes[0].imshow(img1)
    axes[0].set_title("Frame 1")
    axes[0].axis('off')

    # Display Frame 2
    axes[1].imshow(img2)
    axes[1].set_title("Frame 2")
    axes[1].axis('off')

    # Display Predicted Flow (Heatmap Style)
    im_pred = visualize_flow_on_axis(
        axes[2], pred_flow, arrow_step=arrow_step,
        title="Predicted Flow", cmap=cmap
    )
    # Add colorbar for predicted flow
    fig.colorbar(im_pred, ax=axes[2], fraction=0.046, pad=0.04, label='Speed Magnitude (Normalized)')

    # Display GT Flow (if available)
    if has_gt:
        im_gt = visualize_flow_on_axis(
            axes[3], gt_flow, arrow_step=arrow_step,
            title="GT Flow", cmap=cmap
        )
        fig.colorbar(im_gt, ax=axes[3], fraction=0.046, pad=0.04, label='Speed Magnitude (Normalized)')

    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Reserve space for suptitle
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_test_samples(model, test_loader, num_samples=5, save_dir="visualizations/test_samples",
                           arrow_step=10, cmap='jet'):  # Added cmap parameter
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    dataset_size = len(test_loader.dataset)
    if dataset_size == 0:
        print("Test dataset is empty, cannot generate visualization")
        return
    indices = random.sample(range(dataset_size), min(num_samples, dataset_size))

    with torch.no_grad():
        for idx in indices:
            data = test_loader.dataset[idx]
            img1, img2 = data[0].unsqueeze(0).cuda(), data[1].unsqueeze(0).cuda()
            flow_gt = data[2] if len(data) > 2 else None

            flow_preds = model(img1, img2, iters=args.iters)
            pred_flow = flow_preds[-1].squeeze(0)

            losses = None
            if flow_gt is not None:
                if pred_flow.shape != flow_gt.shape:
                    flow_gt_resized = torch.nn.functional.interpolate(
                        flow_gt.unsqueeze(0),
                        size=pred_flow.shape[1:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                else:
                    flow_gt_resized = flow_gt
                losses = calculate_gt_losses(pred_flow.cpu(), flow_gt_resized.cpu())

            save_path = os.path.join(save_dir, f"test_sample_{idx}.png")
            visualize_flow_comparison(
                img1.squeeze(0),
                img2.squeeze(0),
                pred_flow,
                flow_gt,
                save_path=save_path,
                title=f"Test Sample {idx} Flow Visualization",
                arrow_step=arrow_step,
                losses=losses,
                cmap=cmap  # Pass colormap parameter
            )


def train(args):
    visualization_root = os.path.join(args.output_dir, "visualizations")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    log_root = os.path.join(args.output_dir, "log")

    for dir_path in [visualization_root, checkpoint_dir, log_root]:
        os.makedirs(dir_path, exist_ok=True)

    model = nn.DataParallel(AFKAN_PIV(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
    model.cuda()

    train_loader, val_loader = fetch_dataloader(args)
    print(f"Training with {len(train_loader.dataset)} samples, Validation with {len(val_loader.dataset)} samples")

    optimizer, scheduler = fetch_optimizer(args, model)
    scaler = GradScaler(enabled=args.mixed_precision)

    train_logger = Logger(model, scheduler, log_type='train')
    val_logger = Logger(model, scheduler, log_type='val')

    best_epe = float('inf')
    accumulate_steps = args.accumulate_steps
    current_step = 0
    total_train_batches = len(train_loader)

    for epoch in range(args.epochs):
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_metrics = {'epe': 0, '1px': 0, '3px': 0, '5px': 0}

        with tqdm.tqdm(total=len(val_loader), desc=f"Val Epoch {epoch + 1}/{args.epochs}") as pbar:
            for data_blob in val_loader:
                image1, image2, flow, valid = [x.cuda() for x in data_blob]

                if args.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                    image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

                flow_predictions = model(image1, image2, iters=args.iters)
                loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)

                val_loss += loss.item()
                for key in val_metrics:
                    val_metrics[key] += metrics[key]

                pbar.set_postfix(EPE=f"{metrics['epe']:.4f}")
                pbar.update(1)

        # Validation Metrics Logging
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        val_metrics['loss'] = val_loss
        val_logger.write_epoch_dict(val_metrics)

        # Post-validation Visualization
        if (epoch + 1) % 1 == 0:
            print("Generating visualization results on test set...")
            epoch_vis_dir = os.path.join(visualization_root, f"epoch_{epoch + 1}")
            # Optional: Use 'parula' colormap by changing the cmap parameter
            visualize_test_samples(model, val_loader, num_samples=30, save_dir=epoch_vis_dir,
                                   arrow_step=10, cmap='jet')  # Specify colormap here

        # Model Saving
        current_epe = val_metrics['epe']
        if current_epe < best_epe:
            best_epe = current_epe
            save_path = os.path.join(checkpoint_dir, 'afkan_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch + 1} with EPE: {best_epe:.4f}")

        if (epoch + 1) % 200 == 0:
            periodic_path = os.path.join(checkpoint_dir, f'afkan_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), periodic_path)
            print(f"Model saved at epoch {epoch + 1} to {periodic_path}")

        # Training Phase
        model.train()
        train_loss = 0.0
        train_metrics = {'epe': 0, '1px': 0, '3px': 0, '5px': 0}

        with tqdm.tqdm(total=len(train_loader), desc=f"Train Epoch {epoch + 1}/{args.epochs}", unit="batch") as pbar:
            for i_batch, data_blob in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                image1, image2, flow, valid = [x.cuda() for x in data_blob]

                if args.add_noise:
                    stdv = np.random.uniform(0.0, 5.0)
                    image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                    image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

                flow_predictions = model(image1, image2, iters=args.iters)
                loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)

                scaled_loss = loss / accumulate_steps
                scaler.scale(scaled_loss).backward()

                train_loss += loss.item()
                for key in train_metrics:
                    train_metrics[key] += metrics[key]

                pbar.set_postfix(
                    Loss=f"{loss.item():.4f}",
                    EPE=f"{metrics['epe']:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.6f}"
                )
                pbar.update(1)
                current_step += 1

                if current_step % accumulate_steps == 0 or i_batch == total_train_batches - 1:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()
                    model.zero_grad(set_to_none=True)

        # Training Metrics Logging
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        train_metrics['loss'] = train_loss
        train_logger.write_epoch_dict(train_metrics)

    # Final Visualization
    final_vis_dir = os.path.join(visualization_root, "final_results")
    visualize_test_samples(model, val_loader, num_samples=300, save_dir=final_vis_dir,
                           arrow_step=10, cmap='jet')  # Specify colormap here

    train_logger.close()
    val_logger.close()
    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script for AFKAN_PIV Optical Flow Model")

    # Experiment Configuration
    parser.add_argument('--name', default='AFKAN_PIV',
                        help="Name of the experiment, used for logging and identification.")
    parser.add_argument('--stage', default='simdata',
                        help="Specifies the training stage or dataset to use (e.g., 'simdata', 'wallshear').")
    parser.add_argument('--restore_ckpt',
                        help="Path to a checkpoint file (.pth) to resume training from.")
    parser.add_argument('--small', action='store_true',
                        help="If set, uses a smaller version of the model architecture.")
    parser.add_argument('--validation', type=str, nargs='+',
                        help="List of validation datasets to use.")

    # Output Settings
    parser.add_argument('--output_dir', type=str, default='your_output_path',
                        help='Root directory where logs, checkpoints, and visualizations will be saved.')

    # Optimization Parameters
    parser.add_argument('--lr', type=float, default=0.00002,
                        help="Learning rate for the AdamW optimizer.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per iteration (limited by GPU memory).')
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 256],
                        help="Resolution of input images as [height, width].")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                        help="List of GPU IDs to use for training (e.g., 0 1).")
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Enable Automatic Mixed Precision (AMP) to reduce memory usage and speed up training.')
    parser.add_argument('--use_var', type=bool, default=False,
                        help='Whether to manually enforce maximum and minimum displacement constraints.')
    parser.add_argument('--var_max', type=int, default=20,
                        help='Maximum displacement value allowed when use_var is True.')
    parser.add_argument('--var_min', type=int, default=0,
                        help='Minimum displacement value allowed when use_var is True.')
    parser.add_argument('--accumulate_steps', type=int, default=8,
                        help='Number of steps for gradient accumulation. Effective batch size = batch_size * accumulate_steps.')

    # Model Hyperparameters
    parser.add_argument('--iters', type=int, default=1,
                        help="Number of refinement iterations in the flow estimator (GRU updates).")
    parser.add_argument('--wdecay', type=float, default=.00005,
                        help="Weight decay (L2 regularization) factor.")
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help="Epsilon value for the AdamW optimizer to improve numerical stability.")
    parser.add_argument('--clip', type=float, default=1.0,
                        help="Maximum norm for gradient clipping to prevent exploding gradients.")
    parser.add_argument('--dropout', type=float, default=0.0,
                        help="Dropout probability used in the model.")
    parser.add_argument('--dim', type=int, default=128,
                        help="Base feature dimension size in the network.")
    parser.add_argument('--radius', type=int, default=2,
                        help="Radius for the correlation volume or lookup mechanism.")
    parser.add_argument('--num_blocks', type=int, default=3,
                        help="Number of blocks in the feature extraction backbone.")
    parser.add_argument('--block_dims', type=int, nargs='+', default=[64, 128, 256],
                        help="List of channel dimensions for each block in the backbone.")
    parser.add_argument('--initial_dim', type=int, default=64,
                        help="Initial channel dimension before the first block.")
    parser.add_argument('--pretrain', default='resnet34',
                        help="Backbone architecture type (e.g., 'resnet34', 'resnet18').")
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='Exponential weighting factor for sequence loss calculation.')
    parser.add_argument('--add_noise', action='store_true',
                        help="If set, adds Gaussian noise to input images for data augmentation.")
    parser.add_argument('--epochs', type=int, default=300,
                        help='Total number of training epochs.')
    parser.add_argument('--do_flip', action='store_true',
                        help='Enable random horizontal/vertical flipping for data augmentation.')
    parser.add_argument('--data_root', type=str, default='your_data_path',
                        help='Root directory containing the dataset with train/test subfolders.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    train(args)