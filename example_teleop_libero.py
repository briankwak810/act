"""
Example script demonstrating how to use teleoperation with LIBERO.

This script shows how to run evaluation with keyboard teleoperation enabled,
allowing human intervention during policy execution.
"""

import argparse
from imitate_episodes import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LIBERO evaluation with teleoperation')
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--onscreen_render', action='store_true', help='Enable on-screen rendering (required for teleoperation)')
    parser.add_argument('--enable_teleop', action='store_true', help='Enable keyboard teleoperation')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--policy_class', type=str, default='ACT', help='Policy class')
    parser.add_argument('--task_name', type=str, default='libero', help='Task name')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--norm_stats_dir', type=str, default='libero/norm_stats')
    parser.add_argument('--kl_weight', type=int, default=10)
    parser.add_argument('--chunk_size', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--task_indices', type=int, default=0)
    parser.add_argument('--task_names', type=str, default="put the bowl on top of the cabinet")
    
    args = parser.parse_args()
    
    # Ensure onscreen_render is enabled if teleop is requested
    if args.enable_teleop and not args.onscreen_render:
        print("Warning: --onscreen_render is required for teleoperation. Enabling it automatically.")
        args.onscreen_render = True
    
    main(vars(args))

