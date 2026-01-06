"""Script to verify that the Libero dataset is loaded correctly for ACT model.

This script:
1. Loads the dataset using dataset_loader
2. Checks data shapes and types
3. Visualizes example images
4. Displays statistics about the dataset

Usage:
    python libero/verify_dataset.py --repo-id lerobot/libero_spatial --action-horizon 32 --batch-size 4 --task-indices 0 1
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

# PIL for image saving
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available. Image saving will be skipped.")

# Add parent directory to path to import dataset_loader
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Try to import dataset_loader with helpful error message
try:
    from libero.dataset_loader import (
        create_libero_dataloader,
        load_norm_stats,
    )
except ImportError as e:
    print(ImportError)
    raise


def save_images(batch, save_dir: pathlib.Path, batch_idx: int = 0):
    """Save images using PIL."""
    if not HAS_PIL:
        print("  ⚠️  PIL not available, skipping image saving")
        return
    
    save_dir.mkdir(parents=True, exist_ok=True)
    images = batch["image"]  # (batch, num_cam, C, H, W)
    batch_size, num_cam, C, H, W = images.shape
    sample_idx = 0
    
    for cam_idx in range(num_cam):
        img = images[sample_idx, cam_idx].cpu().numpy()
        # Convert from (C, H, W) to (H, W, C) for PIL
        img = np.transpose(img, (1, 2, 0))
        # Clamp to [0, 1] and convert to [0, 255]
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(img)
        save_path = save_dir / f"batch_{batch_idx}_camera_{cam_idx}.png"
        pil_img.save(save_path)
        print(f"  ✅ Saved image to {save_path}")


def print_statistics(batch, batch_idx: int = 0):
    """Print statistics about the batch."""
    print(f"\n{'='*60}")
    print(f"Batch {batch_idx} Statistics")
    print(f"{'='*60}")
    
    # Image statistics
    images = batch["image"]
    print(f"\nImages:")
    print(f"  Shape: {images.shape} (batch, num_cam, C, H, W)")
    print(f"  Dtype: {images.dtype}")
    print(f"  Min: {images.min().item():.4f}, Max: {images.max().item():.4f}")
    print(f"  Mean: {images.mean().item():.4f}, Std: {images.std().item():.4f}")
    
    # Qpos statistics
    qpos = batch["qpos"]
    print(f"\nQpos (Robot State):")
    print(f"  Shape: {qpos.shape} (batch, state_dim)")
    print(f"  Dtype: {qpos.dtype}")
    print(f"  Min: {qpos.min().item():.4f}, Max: {qpos.max().item():.4f}")
    print(f"  Mean: {qpos.mean().item():.4f}, Std: {qpos.std().item():.4f}")
    print(f"  Per-dimension mean: {qpos.mean(dim=0).cpu().numpy()}")
    print(f"  Per-dimension std: {qpos.std(dim=0).cpu().numpy()}")
    
    # Actions statistics
    actions = batch["actions"]
    print(f"\nActions:")
    print(f"  Shape: {actions.shape} (batch, action_horizon, action_dim)")
    print(f"  Dtype: {actions.dtype}")
    print(f"  Min: {actions.min().item():.4f}, Max: {actions.max().item():.4f}")
    print(f"  Mean: {actions.mean().item():.4f}, Std: {actions.std().item():.4f}")
    
    # Is_pad statistics
    if "is_pad" in batch:
        is_pad = batch["is_pad"]
        print(f"\nIs_pad (Padding Mask):")
        print(f"  Shape: {is_pad.shape} (batch, action_horizon)")
        print(f"  Dtype: {is_pad.dtype}")
        num_padded = is_pad.sum().item()
        total = is_pad.numel()
        print(f"  Padded positions: {num_padded}/{total} ({100*num_padded/total:.1f}%)")
        print(f"  Per-sample padding: {is_pad.sum(dim=1).cpu().numpy()}")
    
    print(f"{'='*60}\n")


def verify_shapes(batch, expected_shapes: dict):
    """Verify that batch shapes match expected shapes."""
    print("\nVerifying shapes...")
    all_correct = True
    
    for key, expected_shape in expected_shapes.items():
        if key not in batch:
            print(f"  ❌ Missing key: {key}")
            all_correct = False
            continue
        
        actual_shape = tuple(batch[key].shape)
        if actual_shape == expected_shape:
            print(f"  ✅ {key}: {actual_shape}")
        else:
            print(f"  ❌ {key}: expected {expected_shape}, got {actual_shape}")
            all_correct = False
    
    if all_correct:
        print("  ✅ All shapes are correct!")
    else:
        print("  ❌ Some shapes are incorrect!")
    
    return all_correct


def main():
    parser = argparse.ArgumentParser(
        description="Verify Libero dataset loading for ACT model"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="physical-intelligence/libero",
        help="LeRobot dataset repository ID",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=32,
        help="Action horizon",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for verification",
    )
    parser.add_argument(
        "--model-action-dim",
        type=int,
        default=7,
        help="Model action dimension (7 for Franka arm)",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image resolution [H W]",
    )
    parser.add_argument(
        "--camera-names",
        type=str,
        nargs="+",
        default=["top", "wrist"],
        help="Camera names (e.g., top wrist)",
    )
    parser.add_argument(
        "--task-indices",
        type=int,
        nargs="+",
        default=None,
        help="Task indices to include (e.g., 0 1 2). If not specified, includes all tasks.",
    )
    parser.add_argument(
        "--task-names",
        type=str,
        nargs="+",
        default=None,
        help="Task names (descriptions) to include. Takes precedence over --task-indices if both provided.",
    )
    parser.add_argument(
        "--norm-stats-dir",
        type=str,
        default="libero/norm_stats",
        help="Directory to load norm stats from (optional), default is libero/norm_stats",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=3,
        help="Number of batches to verify",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="libero/verification_output",
        help="Directory to save verification outputs",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Libero Dataset Verification for ACT Model")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Repo ID: {args.repo_id}")
    print(f"  Action Horizon: {args.action_horizon}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Model Action Dim: {args.model_action_dim}")
    print(f"  Image Resolution: {args.image_resolution}")
    print(f"  Camera Names: {args.camera_names}")
    print(f"  Task Indices: {args.task_indices}")
    print(f"  Task Names: {args.task_names}")
    print(f"  Num Batches to Verify: {args.num_batches}")
    print()

    # Load normalization stats if provided
    norm_stats = None
    if args.norm_stats_dir:
        norm_stats_path = pathlib.Path(args.norm_stats_dir)
        if norm_stats_path.exists() and (norm_stats_path / "norm_stats.json").exists():
            print(f"Loading normalization stats from {norm_stats_path}...")
            norm_stats = load_norm_stats(norm_stats_path)
            print("  ✅ Normalization stats loaded")
        else:
            print(f"  ⚠️  Norm stats not found at {norm_stats_path}, will compute on the fly")

    # Create dataloader
    print("\nCreating dataloader...")
    try:
        dataloader = create_libero_dataloader(
            repo_id=args.repo_id,
            action_horizon=args.action_horizon,
            batch_size=args.batch_size,
            model_action_dim=args.model_action_dim,
            image_resolution=tuple(args.image_resolution),
            camera_names=args.camera_names,
            task_indices=args.task_indices,
            task_names=args.task_names,
            norm_stats=norm_stats,
            shuffle=True,
            num_workers=args.num_workers,
        )
        print(f"  ✅ Dataloader created successfully")
        print(f"  Dataset size: {len(dataloader.dataset)} samples")
    except Exception as e:
        print(f"  ❌ Failed to create dataloader: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expected shapes
    batch_size = args.batch_size
    num_cam = len(args.camera_names)
    C = 3  # RGB
    H, W = args.image_resolution
    action_horizon = args.action_horizon
    action_dim = args.model_action_dim
    state_dim = args.model_action_dim  # qpos has same dim as action_dim for Franka

    expected_shapes = {
        "image": (batch_size, num_cam, C, H, W),
        "qpos": (batch_size, state_dim),
        "actions": (batch_size, action_horizon, action_dim),
        "is_pad": (batch_size, action_horizon),
    }

    # Verify batches
    print(f"\nVerifying {args.num_batches} batches...")
    all_verified = True
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break

        print(f"\n{'='*60}")
        print(f"Processing Batch {batch_idx}")
        print(f"{'='*60}")

        # Verify shapes
        shapes_correct = verify_shapes(batch, expected_shapes)
        if not shapes_correct:
            all_verified = False

        # Print statistics
        print_statistics(batch, batch_idx)

        # Save images
        try:
            save_images(batch, output_dir, batch_idx)
        except Exception as e:
            print(f"  ⚠️  Failed to save images for batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    if all_verified:
        print("✅ All batches verified successfully!")
    else:
        print("❌ Some batches had issues. Check the output above.")
    print(f"\nVisualizations saved to: {output_dir}")
    print("=" * 60)

    return 0 if all_verified else 1


if __name__ == "__main__":
    sys.exit(main())

