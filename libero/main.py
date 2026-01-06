import collections
import dataclasses
import logging
import math
import os
import pathlib
import pickle
import sys

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv
import numpy as np
# from openpi_client import image_tools
# from openpi_client import websocket_client_policy as _websocket_client_policy
import torch
from einops import rearrange
import tqdm
import tyro

# ACT imports
from policy import ACTPolicy
from utils import set_seed

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    realtime_visualization: bool = True  # Enable real-time on-screen rendering window

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.realtime_visualization)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        if args.realtime_visualization:
                            env.render()
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    if args.realtime_visualization:
                        env.render()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed, realtime_visualization=False):
    """Initializes and returns the LIBERO environment, along with the task description.
    
    Args:
        task: LIBERO task object
        resolution: Camera resolution
        seed: Random seed
        realtime_visualization: If True, enables real-time on-screen rendering window
    """
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    
    if realtime_visualization:
        # Use ControlEnv with on-screen rendering for real-time visualization
        env_args["has_renderer"] = True
        env_args["has_offscreen_renderer"] = True  # Need True to render Robosuite renderer
        env = ControlEnv(**env_args)
    else:
        # Use OffScreenRenderEnv for saving videos (no window)
        env = OffScreenRenderEnv(**env_args)
    
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


@dataclasses.dataclass
class TestEnvArgs:
    """Arguments for testing LIBERO environment with dummy policy."""
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"  # Task suite name
    task_id: int = 0  # Which task to test (0-indexed)
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize
    num_episodes: int = 3  # Number of test episodes
    
    #################################################################################################################
    # Dummy policy parameters
    #################################################################################################################
    policy_type: str = "random"  # Options: "random", "zero", "dummy" (dummy = LIBERO_DUMMY_ACTION)
    max_steps: int = 100  # Maximum steps per episode
    
    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/test_videos"  # Path to save test videos
    realtime_visualization: bool = True  # Enable real-time on-screen rendering window
    seed: int = 7  # Random seed

def test_libero_env(args: TestEnvArgs) -> None:
    """Test LIBERO environment setup with a dummy policy.
    
    This function verifies that:
    - Environment can be initialized correctly
    - Observations are in expected format
    - Rendering works (both on-screen and off-screen)
    - Actions can be executed
    - Videos can be saved
    
    Args:
        args: TestEnvArgs containing test configuration
    """
    logging.info("=" * 80)
    logging.info("Testing LIBERO Environment with Dummy Policy")
    logging.info("=" * 80)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    
    if args.task_id >= num_tasks_in_suite:
        raise ValueError(f"task_id {args.task_id} is out of range. Task suite has {num_tasks_in_suite} tasks.")
    
    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(f"Number of tasks in suite: {num_tasks_in_suite}")
    
    # Get task
    task = task_suite.get_task(args.task_id)
    task_description = task.language
    logging.info(f"Testing task {args.task_id}: {task_description}")
    
    # Get default LIBERO initial states
    initial_states = task_suite.get_task_init_states(args.task_id)
    logging.info(f"Number of initial states available: {len(initial_states)}")
    
    # Create output directory
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize LIBERO environment
    env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.realtime_visualization)
    logging.info(f"Environment initialized successfully")
    logging.info(f"Environment type: {type(env)}")
    
    # Define dummy policy
    def get_dummy_action(policy_type, obs):
        """Get action from dummy policy."""
        if policy_type == "random":
            # Random action: small random movements
            action = np.random.uniform(-0.1, 0.1, size=6).tolist() + [np.random.choice([-1.0, 1.0])]
        elif policy_type == "zero":
            # Zero action (no movement)
            action = [0.0] * 6 + [0.0]
        elif policy_type == "dummy":
            # LIBERO dummy action
            action = LIBERO_DUMMY_ACTION
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")
        return action
    
    # Run test episodes
    for episode_idx in range(args.num_episodes):
        logging.info(f"\n{'='*80}")
        logging.info(f"Episode {episode_idx + 1}/{args.num_episodes}")
        logging.info(f"{'='*80}")
        
        # Reset environment
        env.reset()
        logging.info("Environment reset")
        
        # Set initial state
        initial_state_idx = episode_idx % len(initial_states)
        obs = env.set_init_state(initial_states[initial_state_idx])
        logging.info(f"Initial state set (using state {initial_state_idx})")
        
        # Check observation format
        logging.info(f"Observation keys: {list(obs.keys())}")
        if "agentview_image" in obs:
            img_shape = obs["agentview_image"].shape
            logging.info(f"agentview_image shape: {img_shape}, dtype: {obs['agentview_image'].dtype}")
        if "robot0_eye_in_hand_image" in obs:
            wrist_shape = obs["robot0_eye_in_hand_image"].shape
            logging.info(f"robot0_eye_in_hand_image shape: {wrist_shape}, dtype: {obs['robot0_eye_in_hand_image'].dtype}")
        if "robot0_eef_pos" in obs:
            logging.info(f"robot0_eef_pos: {obs['robot0_eef_pos']}")
        if "robot0_eef_quat" in obs:
            logging.info(f"robot0_eef_quat: {obs['robot0_eef_quat']}")
        if "robot0_gripper_qpos" in obs:
            logging.info(f"robot0_gripper_qpos: {obs['robot0_gripper_qpos']}")
        
        # Setup for episode
        t = 0
        replay_images = []
        done = False
        
        logging.info(f"Starting episode with {args.policy_type} policy...")
        while t < args.max_steps + args.num_steps_wait and not done:
            try:
                # Wait for objects to stabilize
                if t < args.num_steps_wait:
                    if args.realtime_visualization and hasattr(env, 'env') and hasattr(env.env, 'render'):
                        env.env.render()
                    action = LIBERO_DUMMY_ACTION
                    obs, reward, done, info = env.step(action)
                    t += 1
                    continue
                
                # Get action from dummy policy
                action = get_dummy_action(args.policy_type, obs)
                
                # Save image for video (rotate 180 degrees)
                if "robot0_eye_in_hand_image" in obs:
                    img_rotated = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, :]).copy()
                    replay_images.append(img_rotated)
                
                # Render if enabled (ControlEnv wraps robosuite env, render is on env.env)
                if args.realtime_visualization and hasattr(env, 'env') and hasattr(env.env, 'render'):
                    env.env.render()
                
                # Step environment
                obs, reward, done, info = env.step(action)
                
                # Log progress
                if t % 20 == 0:
                    logging.info(f"  Step {t}: reward={reward}, done={done}")
                    if "agentview_image" in obs:
                        logging.info(f"    Image shape: {obs['agentview_image'].shape}")
                
                t += 1
                
            except Exception as e:
                logging.error(f"Error at step {t}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        logging.info(f"Episode {episode_idx + 1} finished: {t} steps, done={done}")
        
        # Save test video
        if replay_images:
            video_path = pathlib.Path(args.video_out_path) / f"test_task{args.task_id}_ep{episode_idx}_{args.policy_type}.mp4"
            try:
                imageio.mimwrite(
                    video_path,
                    replay_images,
                    fps=10,
                )
                logging.info(f"Saved test video to: {video_path}")
            except Exception as e:
                logging.error(f"Failed to save video: {e}")
    
    logging.info("\n" + "=" * 80)
    logging.info("Environment Test Complete!")
    logging.info("=" * 80)
    logging.info(f"Tested {args.num_episodes} episodes")
    logging.info(f"Videos saved to: {args.video_out_path}")
    logging.info("\nIf everything worked, you can proceed to test with ACT policy.")


@dataclasses.dataclass
class EvalBCArgs:
    """Arguments for evaluating ACT policy on LIBERO."""
    #################################################################################################################
    # ACT Policy parameters
    #################################################################################################################
    ckpt_dir: str  # Directory containing the ACT checkpoint
    ckpt_name: str = "policy_best.ckpt"  # Checkpoint filename
    policy_class: str = "ACT"  # Policy class (ACT or CNNMLP)
    chunk_size: int = 100  # Action chunk size for ACT
    kl_weight: int = 10  # KL weight for ACT
    hidden_dim: int = 512  # Hidden dimension
    dim_feedforward: int = 3200  # Feedforward dimension
    temporal_agg: bool = False  # Enable temporal aggregation
    
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    
    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    realtime_visualization: bool = True  # Enable real-time on-screen rendering window
    resize_size: int = 224  # Image resize size for ACT
    query_frequency: int = None  # How often to query policy (None = use chunk_size)
    
    seed: int = 7  # Random Seed (for reproducibility)


def eval_bc_on_libero(args: EvalBCArgs) -> None:
    """Evaluate ACT policy on LIBERO benchmark tasks.
    
    This function loads an ACT policy checkpoint and evaluates it on LIBERO tasks,
    handling the conversion between LIBERO observation/action format and ACT format.
    """
    # Set random seed
    set_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load policy and stats
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    stats_path = os.path.join(args.ckpt_dir, "dataset_stats.pkl")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    
    # Load normalization stats
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    
    # Infer state and action dimensions from stats
    state_dim = len(stats["qpos_mean"]) if isinstance(stats["qpos_mean"], np.ndarray) else stats["qpos_mean"].shape[0]
    action_dim = len(stats["action_mean"]) if isinstance(stats["action_mean"], np.ndarray) else stats["action_mean"].shape[0]
    
    logging.info(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
    
    # Create policy config
    camera_names = ["agentview", "robot0_eye_in_hand"]  # LIBERO camera names
    policy_config = {
        "lr": 1e-5,  # Not used during eval
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": camera_names,
        "state_dim": state_dim,  # Add state_dim to config
    }
    
    # Load policy
    if args.policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError(f"Policy class {args.policy_class} not implemented")
    
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    logging.info(f"Policy loading status: {loading_status}")
    policy.cuda()
    policy.eval()
    logging.info(f"Loaded policy from: {ckpt_path}")
    
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    
    # Set max steps based on task suite
    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    
    # Set query frequency
    query_frequency = args.query_frequency if args.query_frequency is not None else args.chunk_size
    if args.temporal_agg:
        query_frequency = 1
        num_queries = args.chunk_size
    
    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        
        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        
        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(
            task, LIBERO_ENV_RESOLUTION, args.seed, args.realtime_visualization
        )
        
        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Task {task_id}"):
            logging.info(f"\nTask: {task_description}")
            
            # Reset environment
            env.reset()
            
            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            
            # Setup
            t = 0
            replay_images = []
            action_plan = collections.deque()
            
            # For temporal aggregation
            if args.temporal_agg:
                all_time_actions = torch.zeros([max_steps, max_steps + num_queries, action_dim]).cuda()
            
            logging.info(f"Starting episode {task_episodes + 1}...")
            
            while t < max_steps + args.num_steps_wait:
                try:
                    # Wait for objects to stabilize
                    if t < args.num_steps_wait:
                        if args.realtime_visualization:
                            env.render()
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    
                    # Get and preprocess images
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    
                    # Resize images (try image_tools first, fallback to cv2 or PIL)
                    try:
                        # Try to use openpi_client image_tools if available
                        from openpi_client import image_tools
                        img_resized = image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        wrist_img_resized = image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    except (ImportError, AttributeError, NameError):
                        # Fallback to cv2 or PIL
                        try:
                            import cv2
                            img_resized = cv2.resize(img, (args.resize_size, args.resize_size))
                            wrist_img_resized = cv2.resize(wrist_img, (args.resize_size, args.resize_size))
                        except ImportError:
                            from PIL import Image
                            img_resized = np.array(Image.fromarray(img).resize((args.resize_size, args.resize_size)))
                            wrist_img_resized = np.array(Image.fromarray(wrist_img).resize((args.resize_size, args.resize_size)))
                    
                    # Convert to tensor format: [H, W, C] -> [C, H, W]
                    img_tensor = rearrange(torch.from_numpy(img_resized).float() / 255.0, "h w c -> c h w")
                    wrist_img_tensor = rearrange(torch.from_numpy(wrist_img_resized).float() / 255.0, "h w c -> c h w")
                    
                    # Stack images: [2, C, H, W] where 2 is for agentview and wrist camera
                    curr_image = torch.stack([img_tensor, wrist_img_tensor], dim=0)  # [2, C, H, W]
                    curr_image = curr_image.cuda().unsqueeze(0)  # [1, 2, C, H, W]
                    
                    # Save image for replay video
                    replay_images.append(img_resized.astype(np.uint8))
                    
                    # Get state (qpos) - concatenate eef_pos, axis-angle quat, and gripper
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )
                    
                    # Ensure state matches expected dimension
                    if len(state) != state_dim:
                        if len(state) > state_dim:
                            state = state[:state_dim]
                        else:
                            state = np.pad(state, (0, state_dim - len(state)), mode="constant")
                    
                    qpos = pre_process(state)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)  # [1, state_dim]
                    
                    # Query policy
                    if not action_plan or (t % query_frequency == 0):
                        with torch.inference_mode():
                            if args.policy_class == "ACT":
                                all_actions = policy(qpos, curr_image)  # [1, chunk_size, action_dim]
                                all_actions = all_actions.squeeze(0)  # [chunk_size, action_dim]
                                
                                if args.temporal_agg:
                                    all_time_actions[[t], t : t + num_queries] = all_actions
                                
                                # Add actions to plan
                                for i in range(min(query_frequency, len(all_actions))):
                                    action_plan.append(all_actions[i])
                    
                    # Get action from plan
                    if action_plan:
                        raw_action = action_plan.popleft()
                        
                        if args.temporal_agg and t > 0:
                            # Use temporal aggregation
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            if len(actions_for_curr_step) > 0:
                                k = 0.01
                                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                                exp_weights = exp_weights / exp_weights.sum()
                                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        # Fallback: use dummy action if plan is empty
                        raw_action = torch.zeros(action_dim).cuda()
                    
                    # Post-process action
                    if isinstance(raw_action, torch.Tensor):
                        raw_action_np = raw_action.squeeze().cpu().numpy()
                    else:
                        raw_action_np = raw_action
                    action = post_process(raw_action_np)
                    
                    # Ensure action is 7D (6D arm + 1D gripper) for LIBERO
                    if len(action) != 7:
                        # If action is different size, pad or truncate
                        if len(action) > 7:
                            action = action[:7]
                        else:
                            action = np.pad(action, (0, 7 - len(action)), mode="constant")
                    
                    # Execute action in environment
                    if args.realtime_visualization:
                        env.render()
                    obs, reward, done, info = env.step(action.tolist())
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    
                    t += 1
                    
                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            task_episodes += 1
            total_episodes += 1
            
            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = pathlib.Path(args.video_out_path) / f"rollout_task{task_id}_ep{episode_idx}_{suffix}.mp4"
            imageio.mimwrite(
                video_path,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        
        # Log final results for this task
        logging.info(f"Task {task_id} success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
    
    # Final summary
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total successes: {total_successes}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Uncomment to test environment with dummy policy:
    tyro.cli(test_libero_env)
    
    # Default: run evaluation with websocket client
    # tyro.cli(eval_libero)
