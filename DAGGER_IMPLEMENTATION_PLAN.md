# DAgger Implementation Plan with Keyboard Teleop

## Overview
Implement Dataset Aggregation (DAgger) algorithm where:
1. Policy executes actions in the environment
2. Human expert intervenes via keyboard teleop when policy makes mistakes
3. Corrected actions are collected and aggregated with existing dataset
4. Policy is retrained on aggregated dataset
5. Process repeats iteratively

## Components to Implement

### 1. Keyboard Teleop Interface (`keyboard_teleop.py`)
**Purpose**: Allow human to control robot via keyboard during DAgger episodes

**Key Features**:
- Non-blocking keyboard input (using `keyboard` library or `pynput`)
- Map keyboard keys to robot actions:
  - **Arm control**: WASD/Arrow keys for X/Y translation, QE/PageUp/PageDown for Z
  - **Rotation**: IJKL for pitch/roll, UO for yaw
  - **Gripper**: Space to toggle open/close
- Action scaling/velocity control (hold key = continuous movement)
- Mode toggle: AUTO (policy) vs MANUAL (human control)
- Emergency stop (ESC key)

**Implementation Details**:
- Use `pynput.keyboard` or `keyboard` library for non-blocking input
- Maintain current action state (velocity-based or position-based)
- Convert keyboard inputs to 7D action space: [x, y, z, roll, pitch, yaw, gripper]
- Action smoothing to prevent jerky movements

### 2. DAgger Episode Collection (`collect_dagger_episodes.py`)
**Purpose**: Collect episodes with policy execution + human intervention

**Key Features**:
- Load current policy checkpoint
- Run policy in environment (similar to `eval_bc_libero`)
- At each timestep:
  - Query policy for action
  - Check for keyboard input (non-blocking)
  - If human intervenes: use human action, mark as "corrected"
  - If no intervention: use policy action
- Save episode data with intervention flags

**Data Structure**:
- Observations: images, qpos (same as existing format)
- Actions: actual actions executed (policy or human)
- Metadata: intervention flags, policy actions (for analysis)

**Implementation Details**:
- Integrate keyboard teleop into evaluation loop
- Track when human takes over vs policy
- Save both policy's predicted action and actual executed action
- Handle episode termination (success/failure/timeout)

### 3. Episode Storage (`save_dagger_episode.py`)
**Purpose**: Save DAgger episodes in compatible format

**Key Features**:
- Save episodes in HDF5 format (compatible with existing `EpisodicDataset`)
- Store observations (images, qpos, qvel)
- Store actions (actual executed actions, not policy predictions)
- Store metadata (intervention flags, episode info)
- Support appending to existing dataset

**HDF5 Structure**:
```
episode_N.hdf5:
  - /observations/qpos: (T, 7)
  - /observations/qvel: (T, 7) [optional]
  - /observations/images/agentview: (T, H, W, 3)
  - /observations/images/robot0_eye_in_hand: (T, H, W, 3)
  - /action: (T, 7)
  - /metadata/intervention_flags: (T,) [bool array]
  - /metadata/policy_actions: (T, 7) [optional, for analysis]
  - attrs: sim=True, dagger_iteration=N, episode_id=M
```

### 4. Dataset Aggregation (`aggregate_dagger_dataset.py`)
**Purpose**: Combine new DAgger episodes with existing dataset

**Key Features**:
- Load existing dataset (from initial BC training)
- Load new DAgger episodes
- Merge datasets maintaining episode IDs
- Update normalization statistics
- Create train/val splits

**Implementation Details**:
- Track episode IDs across iterations
- Maintain dataset directory structure
- Recompute normalization stats on aggregated dataset
- Support incremental aggregation (append new episodes)

### 5. DAgger Main Loop (`run_dagger.py`)
**Purpose**: Orchestrate the full DAgger training loop

**Key Features**:
- Iterative training loop:
  1. Evaluate current policy (optional, for monitoring)
  2. Collect N episodes with human intervention
  3. Aggregate new episodes with existing dataset
  4. Retrain policy on aggregated dataset
  5. Save checkpoint
  6. Repeat
- Configuration for:
  - Number of DAgger iterations
  - Episodes per iteration
  - Training epochs per iteration
  - Checkpoint management

**Implementation Details**:
- Use argparse for configuration
- Track DAgger iteration number
- Save checkpoints with iteration info
- Log metrics (success rate, intervention rate, etc.)

### 6. Integration Points

#### Modify `imitate_episodes.py`:
- Add function to load and aggregate datasets
- Support training on aggregated dataset
- Track DAgger iteration in config

#### Modify `libero/dataset_loader.py` (if needed):
- Ensure compatibility with HDF5 format from DAgger episodes
- Support loading mixed datasets (initial + DAgger)

## Implementation Order

### Phase 1: Keyboard Teleop (Foundation)
1. Create `keyboard_teleop.py` with basic keyboard input handling
2. Test keyboard input in isolation
3. Map keys to action space
4. Test action generation

### Phase 2: Episode Collection
1. Create `collect_dagger_episodes.py`
2. Integrate keyboard teleop into evaluation loop
3. Test episode collection with manual control
4. Test episode collection with policy + intervention

### Phase 3: Data Storage
1. Create `save_dagger_episode.py`
2. Test saving episodes in correct format
3. Verify compatibility with existing dataloader

### Phase 4: Dataset Aggregation
1. Create `aggregate_dagger_dataset.py`
2. Test merging datasets
3. Test normalization stat recomputation

### Phase 5: DAgger Loop
1. Create `run_dagger.py`
2. Integrate all components
3. Test full DAgger loop

## Key Design Decisions

### 1. Intervention Strategy
- **Option A**: Human can take over at any timestep (recommended)
- **Option B**: Human only intervenes when policy fails
- **Option C**: Scheduled intervention (e.g., every N steps)

### 2. Action Recording
- **Store actual executed actions** (policy or human) - this is what DAgger requires
- Optionally store policy predictions for analysis
- Store intervention flags for understanding when corrections happened

### 3. Keyboard Mapping
- **Cartesian control** (recommended for LIBERO):
  - WASD: X/Y translation
  - QE: Z translation
  - IJKL: Rotation (pitch/roll)
  - UO: Yaw
  - Space: Gripper toggle
- **Alternative**: Joint space control (more complex)

### 4. Action Space
- LIBERO uses 7D: [x, y, z, roll, pitch, yaw, gripper]
- Actions are delta/velocity commands or absolute positions
- Need to match existing action space format

### 5. Episode Format
- Use existing HDF5 format for compatibility
- Add metadata for DAgger-specific info
- Ensure backward compatibility with existing dataloader

## Dependencies

### New Python Packages:
- `pynput` or `keyboard` - for non-blocking keyboard input
- `threading` - for concurrent keyboard monitoring (if needed)

### Existing Dependencies:
- `h5py` - for episode storage (already used)
- `numpy`, `torch` - for data handling (already used)
- LIBERO environment - already integrated

## Testing Strategy

1. **Unit Tests**:
   - Keyboard input parsing
   - Action generation from keyboard
   - Episode saving/loading

2. **Integration Tests**:
   - Full episode collection with teleop
   - Dataset aggregation
   - Training on aggregated dataset

3. **End-to-End Test**:
   - Run 1-2 DAgger iterations
   - Verify policy improvement

## Configuration Example

```python
dagger_config = {
    'num_iterations': 5,
    'episodes_per_iteration': 10,
    'training_epochs_per_iteration': 100,
    'initial_ckpt': 'policy_best.ckpt',
    'dataset_dir': 'data/dagger_episodes',
    'aggregated_dataset_dir': 'data/dagger_aggregated',
    'ckpt_dir': 'libero/ckpts/dagger',
    'task_name': 'libero_goal_task_04',
    'intervention_mode': 'anytime',  # or 'on_failure', 'scheduled'
}
```

## Future Enhancements

1. **Visual Feedback**: Show policy action vs human action overlay
2. **Intervention Analysis**: Track which states require intervention
3. **Adaptive Intervention**: Learn when to ask for help
4. **Multi-task Support**: DAgger across multiple tasks
5. **Replay Buffer**: Prioritize difficult states for retraining

