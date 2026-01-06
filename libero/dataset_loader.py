'''
create dataset for libero in LeRobot format
'''

"""Standalone script to download LeRobot Libero dataset and create a PyTorch dataloader for ACT model.

This script:
1. Downloads a LeRobot-style libero dataset
2. Creates a PyTorch dataset from it
3. Calculates running normalization statistics for qpos and actions
4. Transforms the dataset into ACT model input specifications (DETRVAE with state_dim=7)
5. Creates a PyTorch DataLoader

Required dependencies:
    pip install lerobot torch numpy einops pillow fsspec gcsfs tqdm

Usage:
    python dataset_loader.py --repo-id lerobot/libero_spatial --action-horizon 32 --batch-size 32 --task-indices 0 1 2
"""

import argparse
import dataclasses
import json
import logging
import pathlib
from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar, runtime_checkable

import einops
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
DataDict = dict
T = TypeVar("T")
S = TypeVar("S")


# ============================================================================
# Normalization Statistics
# ============================================================================


@dataclasses.dataclass
class NormStats:
    """Normalization statistics for a data field."""

    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None  # 1st quantile
    q99: np.ndarray | None = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """Update the running statistics with a batch of vectors."""
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """Compute and return the statistics of the vectors processed so far."""
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


def save_norm_stats(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    stats_dict = {}
    for key, stats in norm_stats.items():
        stats_dict[key] = {
            "mean": stats.mean.tolist(),
            "std": stats.std.tolist(),
            "q01": stats.q01.tolist() if stats.q01 is not None else None,
            "q99": stats.q99.tolist() if stats.q99 is not None else None,
        }

    path.write_text(json.dumps(stats_dict, indent=2))


def load_norm_stats(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")

    data = json.loads(path.read_text())
    norm_stats = {}
    for key, stats in data.items():
        norm_stats[key] = NormStats(
            mean=np.array(stats["mean"]),
            std=np.array(stats["std"]),
            q01=np.array(stats["q01"]) if stats["q01"] is not None else None,
            q99=np.array(stats["q99"]) if stats["q99"] is not None else None,
        )
    return norm_stats


# ============================================================================
# Image Utilities
# ============================================================================


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Resize images to target height/width with padding using PIL.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> np.ndarray:
    """Resize an image to target height/width without distortion by padding with zeros."""
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)  # No need to resize if already correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return np.array(zero_image)


# ============================================================================
# Data Transforms
# ============================================================================


@runtime_checkable
class DataTransformFn(Protocol):
    """Protocol for data transformation functions."""

    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data."""
        ...


@dataclasses.dataclass
class CompositeTransform:
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def _parse_image(image) -> np.ndarray:
    """Parse image from LeRobot format to (H, W, C) uint8."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass
class RepackTransform:
    """Repacks an input dictionary into a new dictionary using flattened paths.

    The structure dict maps new keys to old flattened paths (using '/' separator).
    Example: {"observation/image": "image"} maps data["observation"]["image"] to result["image"]
    """

    structure: dict

    def __call__(self, data: DataDict) -> DataDict:
        def _flatten_dict(d, parent_key="", sep="/"):
            """Flatten a nested dictionary using the separator."""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        def _get_nested_value(d, path, sep="/"):
            """Get a value from a nested dict using a path."""
            parts = path.split(sep)
            value = d
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value

        # For simple key remapping (like in Libero config), just remap the keys
        # The structure maps new_key -> old_path
        result = {}
        for new_key, old_path in self.structure.items():
            # Try to get value using nested path
            value = _get_nested_value(data, old_path)
            if value is not None:
                # If new_key contains '/', we need to create nested structure
                if "/" in new_key:
                    parts = new_key.split("/")
                    d = result
                    for part in parts[:-1]:
                        if part not in d:
                            d[part] = {}
                        d = d[part]
                    d[parts[-1]] = value
                else:
                    result[new_key] = value

        return result


@dataclasses.dataclass
class ACTInputs:
    """Convert LeRobot Libero data to ACT model input format.
    
    ACT model expects:
    - qpos: (state_dim,) - robot state, typically 7 for Franka arm
    - image: (num_cam, C, H, W) - images from multiple cameras
    - actions: (action_horizon, action_dim) - action sequence, action_dim=7 for Franka
    """

    camera_names: list[str]

    def __call__(self, data: DataDict) -> DataDict:
        # Helper to get nested value from dict
        def _get_nested_value(d, path, sep="/"):
            parts = path.split(sep)
            value = d
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value
        
        # Parse images from different cameras
        images = []
        for cam_name in self.camera_names:
            img = None
            # Try nested structure first (from RepackTransform: data["observation"]["image"])
            if "observation" in data and isinstance(data["observation"], dict):
                if cam_name == "top" and "image" in data["observation"]:
                    img = _parse_image(data["observation"]["image"])
                elif cam_name == "wrist" and "wrist_image" in data["observation"]:
                    img = _parse_image(data["observation"]["wrist_image"])
                elif cam_name in data["observation"]:
                    img = _parse_image(data["observation"][cam_name])
            
            # Fallback to flat keys (if RepackTransform wasn't used)
            if img is None:
                if f"observation/{cam_name}" in data:
                    img = _parse_image(data[f"observation/{cam_name}"])
                elif "observation/image" in data and cam_name == "top":
                    img = _parse_image(data["observation/image"])
                elif "observation/wrist_image" in data and cam_name == "wrist":
                    img = _parse_image(data["observation/wrist_image"])
            
            if img is None:
                raise ValueError(f"Camera {cam_name} not found in data. Available keys: {list(data.keys())}")
            images.append(img)
        
        # Stack images: (num_cam, H, W, C)
        images_array = np.stack(images, axis=0)
        
        # Convert to (num_cam, C, H, W) format expected by ACT
        images_array = einops.rearrange(images_array, "n h w c -> n c h w")
        
        # Get qpos/state - handle both nested and flat structures
        qpos = None
        if "observation" in data and isinstance(data["observation"], dict) and "state" in data["observation"]:
            qpos = data["observation"]["state"]
        elif "observation/state" in data:
            qpos = data["observation/state"]
        else:
            raise ValueError(f"State not found in data. Available keys: {list(data.keys())}")
        
        inputs = {
            "qpos": qpos,  # qpos is the robot state
            "image": images_array,
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        return inputs


@dataclasses.dataclass
class ResizeImages:
    """Resize images to target height and width.
    
    Input: (num_cam, C, H, W)
    Output: (num_cam, C, height, width)
    """

    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        # data["image"] is (num_cam, C, H, W)
        images = data["image"]
        num_cam, C, H, W = images.shape
        
        # Convert to (num_cam, H, W, C) for resize_with_pad
        images = einops.rearrange(images, "n c h w -> n h w c")
        
        # Resize each camera image
        resized_images = []
        for i in range(num_cam):
            resized = resize_with_pad(images[i:i+1], self.height, self.width)
            resized_images.append(resized[0])  # Remove batch dimension
        
        # Stack and convert back to (num_cam, C, H, W)
        resized_images = np.stack(resized_images, axis=0)
        resized_images = einops.rearrange(resized_images, "n h w c -> n c h w")
        
        data["image"] = resized_images
        return data


@dataclasses.dataclass
class PadStatesAndActions:
    """Zero-pads qpos and actions to the model action dimension (7 for Franka arm)."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
            current_dim = x.shape[axis]
            if current_dim < target_dim:
                pad_width = [(0, 0)] * len(x.shape)
                pad_width[axis] = (0, target_dim - current_dim)
                return np.pad(x, pad_width, constant_values=value)
            elif current_dim > target_dim:
                # Truncate if larger
                slices = [slice(None)] * len(x.shape)
                slices[axis] = slice(0, target_dim)
                return x[tuple(slices)]
            return x

        data["qpos"] = pad_to_dim(data["qpos"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


@dataclasses.dataclass
class CreateIsPad:
    """Create is_pad mask for action sequences.
    
    Creates a boolean mask where True indicates padding positions.
    For LeRobot datasets, actions are already padded to action_horizon,
    so we check for zero actions to determine padding.
    """

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" in data:
            actions = data["actions"]
            # Convert to numpy if it's a torch tensor
            if isinstance(actions, torch.Tensor):
                actions = actions.numpy()
            # Check if action is all zeros (padded)
            # For action sequence (action_horizon, action_dim)
            is_pad = np.all(actions == 0, axis=-1)  # (action_horizon,)
            data["is_pad"] = is_pad.astype(bool)
        return data


@dataclasses.dataclass
class Normalize:
    """Normalize qpos and actions using normalization statistics."""

    norm_stats: dict[str, NormStats] | None
    use_quantiles: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Normalize qpos (robot state)
        if "qpos" in data and "qpos" in self.norm_stats:
            stats = self.norm_stats["qpos"]
            if self.use_quantiles:
                assert stats.q01 is not None and stats.q99 is not None
                q01, q99 = stats.q01, stats.q99
                dim = min(q01.shape[-1], data["qpos"].shape[-1])
                data["qpos"][..., :dim] = (data["qpos"][..., :dim] - q01[..., :dim]) / (
                    q99[..., :dim] - q01[..., :dim] + 1e-6
                ) * 2.0 - 1.0
            else:
                mean, std = stats.mean, stats.std
                dim = min(mean.shape[-1], data["qpos"].shape[-1])
                data["qpos"][..., :dim] = (data["qpos"][..., :dim] - mean[..., :dim]) / (std[..., :dim] + 1e-6)

        # Normalize actions
        if "actions" in data and "actions" in self.norm_stats:
            stats = self.norm_stats["actions"]
            if self.use_quantiles:
                assert stats.q01 is not None and stats.q99 is not None
                q01, q99 = stats.q01, stats.q99
                dim = min(q01.shape[-1], data["actions"].shape[-1])
                data["actions"][..., :dim] = (data["actions"][..., :dim] - q01[..., :dim]) / (
                    q99[..., :dim] - q01[..., :dim] + 1e-6
                ) * 2.0 - 1.0
            else:
                mean, std = stats.mean, stats.std
                dim = min(mean.shape[-1], data["actions"].shape[-1])
                data["actions"][..., :dim] = (data["actions"][..., :dim] - mean[..., :dim]) / (std[..., :dim] + 1e-6)

        return data




# ============================================================================
# Dataset and DataLoader
# ============================================================================


class TransformedDataset(Dataset):
    """Dataset that applies transforms to samples."""

    def __init__(self, dataset: Dataset, transforms: Sequence[DataTransformFn]):
        self._dataset = dataset
        self._transform = CompositeTransform(transforms)

    def __getitem__(self, index: int) -> dict:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


def _collate_fn(items):
    """Collate batch elements into batched numpy arrays.
    
    Returns:
        Dictionary with:
        - image: (batch, num_cam, C, H, W) as torch.Tensor
        - qpos: (batch, state_dim) as torch.Tensor
        - actions: (batch, action_horizon, action_dim) as torch.Tensor
        - is_pad: (batch, action_horizon) as torch.Tensor
    """
    batch = {}
    for k in items[0].keys():
        if k == "image":
            # Convert to torch tensor and normalize to [0, 1]
            images = [torch.from_numpy(np.asarray(item[k])).float() / 255.0 for item in items]
            batch[k] = torch.stack(images, dim=0).cuda()  # (batch, num_cam, C, H, W)
        elif k in ["qpos", "actions"]:
            tensors = [torch.from_numpy(np.asarray(item[k])).float() for item in items]
            batch[k] = torch.stack(tensors, dim=0).cuda()
        elif k == "is_pad":
            tensors = [torch.from_numpy(np.asarray(item[k])).bool() for item in items]
            batch[k] = torch.stack(tensors, dim=0).cuda()
        else:
            batch[k] = [item[k] for item in items]
    return batch


def create_libero_dataset(
    repo_id: str,
    action_horizon: int,
    task_indices: list[int] | None = None,
    task_names: list[str] | None = None,
) -> Dataset:
    """Create a LeRobot Libero dataset filtered by task indices or task names.

    Args:
        repo_id: The LeRobot dataset repository ID (e.g., "lerobot/libero_spatial")
        action_horizon: The action horizon for the dataset
        task_indices: List of task indices to include. If None, includes all tasks.
        task_names: List of task names (descriptions) to include. If provided, will be
                    converted to task_indices. Takes precedence over task_indices if both provided.

    Returns:
        A PyTorch Dataset
    """
    
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            "actions": [t / dataset_meta.fps for t in range(action_horizon)]
        },
    )
    # need downgrade to lerobot 1.0 for physicalintelligence-libero dataset (version issue)

    # Convert task_names to task_indices if provided
    if task_names is not None:
        task_indices_from_names = []
        for task_name in task_names:
            if task_name in dataset_meta.task_to_task_index:
                task_indices_from_names.append(dataset_meta.task_to_task_index[task_name])
            else:
                logger.warning(f"Task name '{task_name}' not found in dataset. Available tasks: {list(dataset_meta.task_to_task_index.keys())[:5]}...")
        if task_indices_from_names:
            task_indices = task_indices_from_names
            logger.info(f"Converted task_names to task_indices: {task_indices}")
        else:
            logger.warning("No valid task_names found, using all tasks")
            task_indices = None

    # Filter by task indices if specified
    if task_indices is not None:
        task_indices_set = set(task_indices)
        
        class FilterByTask:
            def __call__(self, data: DataDict) -> DataDict:
                if "task_index" not in data:
                    # If no task_index, skip this sample
                    logger.warning("No task_index found in data")
                    return None
                task_index = int(data["task_index"])
                if task_index not in task_indices_set:
                    return None
                return data
        
        # Apply filter and remove None samples
        class FilteredDataset(Dataset):
            def __init__(self, base_dataset, filter_fn):
                self._dataset = base_dataset
                self._filter = filter_fn
                self._valid_indices = []
                
                # Try to access task_index directly from underlying HuggingFace dataset
                if hasattr(base_dataset, 'hf_dataset') and 'task_index' in base_dataset.hf_dataset.column_names:
                    logger.info("Using optimized filtering: accessing task_index directly from dataset...")
                    task_index_col = base_dataset.hf_dataset["task_index"]
                    # Convert Column to list if needed (for newer datasets versions)
                    if hasattr(task_index_col, '__iter__') and not isinstance(task_index_col, (list, tuple)):
                        task_index_col = list(task_index_col)
                    
                    for i, task_idx_tensor in enumerate(task_index_col):
                        # Extract value from tensor if needed
                        task_idx = int(task_idx_tensor.item() if hasattr(task_idx_tensor, 'item') else task_idx_tensor)
                        if task_idx in task_indices_set:
                            self._valid_indices.append(i)
                else:
                    logger.warning("Optimized filtering not available, using slower method...")
                    for i in range(len(base_dataset)):
                        sample = base_dataset[i]
                        if filter_fn(sample) is not None:
                            self._valid_indices.append(i)
                
                logger.info(f"Filtered dataset to {len(self._valid_indices)} samples from tasks {task_indices}")
            
            def __getitem__(self, index: int) -> dict:
                original_index = self._valid_indices[index]
                return self._dataset[original_index]
            
            def __len__(self) -> int:
                return len(self._valid_indices)
        
        dataset = FilteredDataset(dataset, FilterByTask())
        logger.info(f"Filtered dataset to {len(dataset)} samples from tasks {task_indices}")

    return dataset


def transform_libero_dataset(
    dataset: Dataset,
    model_action_dim: int,
    image_resolution: tuple[int, int] = (224, 224),
    camera_names: list[str] = None,
    norm_stats: dict[str, NormStats] | None = None,
    use_quantile_norm: bool = False,
) -> Dataset:
    """Transform Libero dataset to ACT model input format.

    Args:
        dataset: The input dataset
        model_action_dim: The model's action dimension (7 for Franka arm)
        image_resolution: Target image resolution (height, width)
        camera_names: List of camera names to use (e.g., ["top", "wrist"])
        norm_stats: Normalization statistics for qpos and actions
        use_quantile_norm: Whether to use quantile normalization

    Returns:
        Transformed dataset
    """
    if camera_names is None:
        camera_names = ["top", "wrist"]  # Default to top and wrist cameras
    
    transforms_list = []

    # Repack transform: remap keys from LeRobot format
    # Maps new_key -> old_key (dataset returns 'image', 'wrist_image', 'state', not 'observation/...')
    repack_structure = {
        "observation/state": "state",
        "actions": "actions",
    }
    # Add camera image keys
    for cam_name in camera_names:
        if cam_name == "top":
            repack_structure["observation/image"] = "image"
        elif cam_name == "wrist":
            repack_structure["observation/wrist_image"] = "wrist_image"
        else:
            repack_structure[f"observation/{cam_name}"] = cam_name
    
    transforms_list.append(RepackTransform(repack_structure))

    # Data transform: convert to ACT format
    transforms_list.append(ACTInputs(camera_names=camera_names))

    # Resize images
    transforms_list.append(ResizeImages(image_resolution[0], image_resolution[1]))

    # Normalize qpos and actions
    if norm_stats:
        transforms_list.append(Normalize(norm_stats, use_quantiles=use_quantile_norm))

    # Pad states and actions to model_action_dim
    transforms_list.append(PadStatesAndActions(model_action_dim))
    
    # Create is_pad mask
    transforms_list.append(CreateIsPad())

    return TransformedDataset(dataset, transforms_list)


def compute_norm_stats(
    dataset: Dataset,
    camera_names: list[str] = None,
    batch_size: int = 32,
    max_frames: int | None = None,
    num_workers: int = 0,
    image_resolution: tuple[int, int] = None,
) -> tuple[dict[str, NormStats], Dataset]:
    """Compute normalization statistics from a dataset for qpos and actions.

    Args:
        dataset: The dataset to compute stats from
        camera_names: List of camera names to use
        batch_size: Batch size for processing
        max_frames: Maximum number of frames to process (None for all)
        num_workers: Number of worker processes
        image_resolution: Image resolution (height, width) for resizing

    Returns:
        Tuple of (normalization statistics dict, preprocessed dataset)
        The preprocessed dataset can be reused to avoid redundant transformations
    """
    if camera_names is None:
        camera_names = ["top"]
    
    # Create a simplified transform that doesn't normalize
    class RemoveStrings:
        def __call__(self, data: DataDict) -> DataDict:
            return {k: v for k, v in data.items() if not isinstance(v, str)}

    # Apply transforms up to (but not including) normalization
    repack_structure = {
        "observation/state": "state",
        "actions": "actions",
    }
    for cam_name in camera_names:
        if cam_name == "top":
            repack_structure["observation/image"] = "image"
        elif cam_name == "wrist":
            repack_structure["observation/wrist_image"] = "wrist_image"
        else:
            repack_structure[f"observation/{cam_name}"] = cam_name
    
    repack_transform = RepackTransform(repack_structure)

    transforms_list = [
        repack_transform,
        ACTInputs(camera_names=camera_names),
    ]
    
    # Add resize if image_resolution is provided
    if image_resolution is not None:
        transforms_list.append(ResizeImages(image_resolution[0], image_resolution[1]))
    
    transforms_list.append(RemoveStrings())

    stats_dataset = TransformedDataset(dataset, transforms_list)

    # Create dataloader
    if max_frames is not None and max_frames < len(stats_dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(stats_dataset) // batch_size
        shuffle = False

    data_loader = DataLoader(
        stats_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )

    # Compute stats for qpos and actions
    keys = ["qpos", "actions"]
    stats = {key: RunningStats() for key in keys}

    import tqdm

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            if key in batch:
                # Convert torch tensor to numpy
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].numpy()
                stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats[key].get_statistics() for key in keys if key in stats}
    return norm_stats, stats_dataset


def create_libero_dataloader(
    repo_id: str,
    action_horizon: int,
    batch_size: int,
    model_action_dim: int = 7,
    image_resolution: tuple[int, int] = (224, 224),
    camera_names: list[str] = None,
    task_indices: list[int] | None = None,
    task_names: list[str] | None = None,
    norm_stats: dict[str, NormStats] | None = None,
    norm_stats_dir: str = "libero/norm_stats",
    use_quantile_norm: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a complete PyTorch DataLoader for Libero dataset compatible with ACT model.

    Args:
        repo_id: LeRobot dataset repository ID
        action_horizon: Action horizon for the dataset
        batch_size: Batch size
        model_action_dim: Model action dimension (7 for Franka arm)
        image_resolution: Target image resolution (height, width)
        camera_names: List of camera names to use (e.g., ["top", "wrist"])
        task_indices: List of task indices to include. If None, includes all tasks.
        task_names: List of task names (descriptions) to include. Takes precedence over task_indices.
        norm_stats: Normalization statistics (will be computed if None)
        use_quantile_norm: Whether to use quantile normalization
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes

    Returns:
        PyTorch DataLoader
    """
    if camera_names is None:
        camera_names = ["top"]
    
    # Create base dataset
    base_dataset = create_libero_dataset(repo_id, action_horizon, task_indices=task_indices, task_names=task_names)

    # Compute norm stats if not provided, and get preprocessed dataset
    if norm_stats is None:
        logger.info("Computing normalization statistics...")
        norm_stats, preprocessed_dataset = compute_norm_stats(
            base_dataset, 
            camera_names=camera_names,
            batch_size=batch_size, 
            num_workers=num_workers,
            image_resolution=image_resolution,
        )
        logger.info("Normalization statistics computed.")
        
        # Save norm stats to file
        norm_stats_path = pathlib.Path(norm_stats_dir)
        if norm_stats_path.exists():
            logger.info(f"Saving norm stats to {norm_stats_path}...")
            save_norm_stats(norm_stats_path, norm_stats)
        
        # Reuse the preprocessed dataset and add remaining transforms
        # (normalization, padding, is_pad)
        remaining_transforms = []
        if norm_stats:
            remaining_transforms.append(Normalize(norm_stats, use_quantiles=use_quantile_norm))
        remaining_transforms.append(PadStatesAndActions(model_action_dim))
        remaining_transforms.append(CreateIsPad())
        
        dataset = TransformedDataset(preprocessed_dataset, remaining_transforms)
    else:
        # Transform dataset from scratch if norm_stats already provided
        dataset = transform_libero_dataset(
            base_dataset,
            model_action_dim=model_action_dim,
            image_resolution=image_resolution,
            camera_names=camera_names,
            norm_stats=norm_stats,
            use_quantile_norm=use_quantile_norm,
        )

    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        drop_last=True,
    )


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Create PyTorch DataLoader for LeRobot Libero dataset (ACT model)")
    parser.add_argument("--repo-id", type=str, default="physical-intelligence/libero", help="LeRobot dataset repo ID")
    parser.add_argument("--action-horizon", type=int, default=32, help="Action horizon")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--model-action-dim", type=int, default=7, help="Model action dimension (7 for Franka arm)")
    parser.add_argument("--image-resolution", type=int, nargs=2, default=[224, 224], help="Image resolution [H W]")
    parser.add_argument("--camera-names", type=str, nargs="+", default=["top"], help="Camera names (e.g., top wrist)")
    parser.add_argument("--task-indices", type=int, nargs="+", default=None, help="Task indices to include (e.g., 0 1 2). If not specified, includes all tasks.")
    parser.add_argument("--task-names", type=str, nargs="+", default=None, help="Task names (descriptions) to include. Takes precedence over --task-indices if both provided.")
    parser.add_argument("--norm-stats-dir", type=str, default=None, help="Directory to load/save norm stats")
    parser.add_argument("--use-quantile-norm", action="store_true", help="Use quantile normalization")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of worker processes")
    parser.add_argument("--compute-stats-only", action="store_true", help="Only compute and save stats, don't create dataloader")

    args = parser.parse_args()

    # Create dataset
    logger.info(f"Creating dataset from {args.repo_id}...")
    dataset = create_libero_dataset(
        args.repo_id, 
        args.action_horizon, 
        task_indices=args.task_indices,
        task_names=args.task_names
    )

    # Load or compute norm stats
    norm_stats = None
    if args.norm_stats_dir:
        norm_stats_path = pathlib.Path(args.norm_stats_dir)
        if norm_stats_path.exists() and (norm_stats_path / "norm_stats.json").exists():
            logger.info(f"Loading norm stats from {norm_stats_path}...")
            norm_stats = load_norm_stats(norm_stats_path)
        else:
            logger.info("Computing normalization statistics...")
            norm_stats, _ = compute_norm_stats(
                dataset, 
                camera_names=args.camera_names,
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                image_resolution=tuple(args.image_resolution),
            )
            logger.info(f"Saving norm stats to {norm_stats_path}...")
            save_norm_stats(norm_stats_path, norm_stats)
    else:
        logger.info("Computing normalization statistics...")
        norm_stats, _ = compute_norm_stats(
            dataset, 
            camera_names=args.camera_names,
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            image_resolution=tuple(args.image_resolution),
        )

    if args.compute_stats_only:
        logger.info("Stats computed. Exiting.")
        return

    # Create dataloader
    logger.info("Creating DataLoader...")
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
        use_quantile_norm=args.use_quantile_norm,
        num_workers=args.num_workers,
    )

    # Test the dataloader
    logger.info("Testing DataLoader...")
    for i, batch in enumerate(dataloader):
        logger.info(f"Batch {i}:")
        logger.info(f"  Image shape: {batch['image'].shape}")  # (batch, num_cam, C, H, W)
        logger.info(f"  Qpos shape: {batch['qpos'].shape}")  # (batch, state_dim)
        if "actions" in batch:
            logger.info(f"  Actions shape: {batch['actions'].shape}")  # (batch, action_horizon, action_dim)
        if "is_pad" in batch:
            logger.info(f"  Is_pad shape: {batch['is_pad'].shape}")  # (batch, action_horizon)
        if i >= 2:  # Just test a few batches
            break

    logger.info("DataLoader created successfully!")


if __name__ == "__main__":
    main()

