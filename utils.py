import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader, random_split
from libero.dataset_loader import (
    load_norm_stats, create_libero_dataloader, create_libero_dataset, 
    transform_libero_dataset, compute_norm_stats, _collate_fn,
    Normalize, PadStatesAndActions, CreateIsPad, TransformedDataset, save_norm_stats
)
import pathlib

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def load_data_from_lerobot(config, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nLoading data from Lerobot\n')
    norm_stats_dir = config['norm_stats_dir']
    norm_stats = None
    if norm_stats_dir:
        norm_stats_path = pathlib.Path(norm_stats_dir)
        if norm_stats_path.exists() and (norm_stats_path / "norm_stats.json").exists():
            print(f"Loading normalization stats from {norm_stats_path}...")
            norm_stats = load_norm_stats(norm_stats_path)
            print("  ✅ Normalization stats loaded")
        else:
            print(f"  ⚠️ Norm stats not found at {norm_stats_path}, will compute on the fly")

    # Convert task_names to list if it's a string
    task_names = config.get('task_names')
    if task_names is not None and isinstance(task_names, str):
        task_names = [task_names]
    
    task_indices = config.get('task_indices')
    if task_indices is not None and not isinstance(task_indices, (list, tuple)):
        task_indices = [task_indices] if task_indices != 0 else None

    # Get action_horizon from chunk_size (num_queries) to match model expectations
    # 32 intended
    # action_horizon = config.get('policy_config', {}).get('num_queries', 100)
    # if action_horizon is None:
    #     action_horizon = 100  # default fallback

    # Create dataset
    print("\nCreating dataset...")
    try:
        base_dataset = create_libero_dataset(
            repo_id="physical-intelligence/libero",
            action_horizon=32,
            task_indices=task_indices,
            task_names=task_names,
        )
        print(f"  ✅ Base dataset created: {len(base_dataset)} samples")
    except Exception as e:
        print(f"  ❌ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Compute norm stats if not provided
    if norm_stats is None:
        print("\nComputing normalization statistics...")
        try:
            norm_stats, preprocessed_dataset = compute_norm_stats(
                base_dataset,
                camera_names=config['camera_names'],
                batch_size=batch_size_train,
                num_workers=4,
                image_resolution=[224, 224],
            )
            print("  ✅ Normalization statistics computed")
            
            # Save norm stats if directory is provided
            if norm_stats_dir:
                norm_stats_path = pathlib.Path(norm_stats_dir)
                if norm_stats_path.exists():
                    print(f"Saving norm stats to {norm_stats_path}...")
                    save_norm_stats(norm_stats_path, norm_stats)
            
            # Transform dataset
            remaining_transforms = []
            remaining_transforms.append(Normalize(norm_stats, use_quantiles=False))
            remaining_transforms.append(PadStatesAndActions(config['state_dim']))
            remaining_transforms.append(CreateIsPad())
            dataset = TransformedDataset(preprocessed_dataset, remaining_transforms)
        except Exception as e:
            print(f"  ❌ Failed to compute norm stats: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        # Transform dataset from scratch if norm_stats already provided
        print("\nTransforming dataset with provided norm stats...")
        try:
            dataset = transform_libero_dataset(
                base_dataset,
                model_action_dim=config['state_dim'],
                image_resolution=[224, 224],
                camera_names=config['camera_names'],
                norm_stats=norm_stats,
                use_quantile_norm=False,
            )
            print("  ✅ Dataset transformed")
        except Exception as e:
            print(f"  ❌ Failed to transform dataset: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Split dataset into train and validation
    print("\nSplitting dataset into train/validation...")
    train_ratio = 0.8
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"  ✅ Dataset split: {train_size} train, {val_size} validation samples")

    # Create dataloaders
    print("\nCreating dataloaders...")
    try:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=0,
            collate_fn=_collate_fn,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=0,
            collate_fn=_collate_fn,
            drop_last=True,
        )
        print(f"  ✅ Dataloaders created successfully")
        print(f"  Train dataset size: {len(train_dataset)} samples")
        print(f"  Val dataset size: {len(val_dataset)} samples")
    except Exception as e:
        print(f"  ❌ Failed to create dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return train_dataloader, val_dataloader

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
