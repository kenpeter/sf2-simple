#!/usr/bin/env python3
"""
🥊 Street Fighter RL Training - Ray RLlib Version
Massively scalable PPO training - can handle 100s to 1000s of parallel environments

Usage:
    # Start with 128 environments (16 workers × 8 envs each):
    python train.py --num-workers 16 --num-envs-per-worker 8

    # Scale up to 512 environments:
    python train.py --num-workers 32 --num-envs-per-worker 16

    # With GPU:
    python train.py --num-gpus 1 --num-workers 16

    # Resume training:
    python train.py --resume ~/ray_results/PPO_xxx/checkpoint_000100
"""

import os
import argparse
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.atari_wrappers import FrameStack
from wrapper import StreetFighter

# Create directories
os.makedirs("train", exist_ok=True)
os.makedirs("logs", exist_ok=True)

print("🥊 Street Fighter RL Training - Ray RLlib (Massively Scalable)")


def env_creator(env_config):
    """
    Create Street Fighter environment for RLlib
    RLlib calls this function to create each parallel environment
    """
    env = StreetFighter()

    # Apply frame stacking
    frame_stack = env_config.get("frame_stack", 4)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)

    return env


class WinRateCallback(tune.Callback):
    """
    Custom callback to track win rate and training progress
    """

    def on_trial_result(self, iteration, trials, trial, result, **info):
        """Called after each training iteration"""
        timesteps = result.get("timesteps_total", 0)
        episodes = result.get("episodes_total", 0)
        mean_reward = result.get("episode_reward_mean", 0)

        # Get win rate from custom metrics
        custom_metrics = result.get("custom_metrics", {})
        agent_won_mean = custom_metrics.get("agent_won_mean", None)

        # Print progress
        print(f"\n📊 Iteration {iteration}:")
        print(f"   Timesteps: {timesteps:,}")
        print(f"   Episodes: {episodes:,}")
        print(f"   Mean Reward: {mean_reward:.2f}")

        if agent_won_mean is not None:
            win_rate = agent_won_mean * 100
            print(f"   Win Rate: {win_rate:.1f}%")


def main():
    """
    Main function - Configure and run RLlib training
    """
    parser = argparse.ArgumentParser(description="Street Fighter RL Training with RLlib")

    # === SCALABILITY PARAMETERS (Most Important!) ===
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel rollout workers. Each worker runs multiple environments. "
             "More workers = more parallel data collection. Recommended: num_cpus - 2"
    )
    parser.add_argument(
        "--num-envs-per-worker",
        type=int,
        default=8,
        help="Environments per worker. TOTAL ENVS = num_workers × num_envs_per_worker. "
             "With 16 workers × 8 envs = 128 total parallel environments!"
    )
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=0,
        help="GPUs for training the neural network. Use 1 if you have a GPU. "
             "Can use fractional values like 0.5 to share GPU"
    )

    # === TRAINING BATCH SIZES ===
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4096,
        help="Training batch size (timesteps collected before each update). "
             "Larger = more stable, smaller = faster iterations. Default: 4096"
    )
    parser.add_argument(
        "--sgd-minibatch-size",
        type=int,
        default=512,
        help="Minibatch size for SGD optimization. Default: 512"
    )
    parser.add_argument(
        "--rollout-fragment-length",
        type=int,
        default=256,
        help="Timesteps each worker collects before sending to learner. Default: 256"
    )

    # === PPO HYPERPARAMETERS ===
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lambda", type=float, default=0.95, dest="lambda_", help="GAE lambda")
    parser.add_argument("--clip-param", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--entropy-coeff", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--num-sgd-iter", type=int, default=10, help="SGD epochs per update")

    # === ENVIRONMENT ===
    parser.add_argument("--frame-stack", type=int, default=4, help="Frames to stack")

    # === TRAINING DURATION ===
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=10_000_000,
        help="Total timesteps to train. Default: 10 million"
    )

    # === CHECKPOINTING ===
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Save every N iterations")
    parser.add_argument("--checkpoint-dir", type=str, default="~/ray_results", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    args = parser.parse_args()

    # Calculate total environments
    total_envs = args.num_workers * args.num_envs_per_worker

    print("\n" + "="*80)
    print("🥊 RAY RLLIB STREET FIGHTER - MASSIVELY SCALABLE TRAINING")
    print("="*80)
    print(f"\n🚀 PARALLELIZATION:")
    print(f"   Workers: {args.num_workers}")
    print(f"   Environments per worker: {args.num_envs_per_worker}")
    print(f"   ⚡ TOTAL PARALLEL ENVIRONMENTS: {total_envs} ⚡")
    print(f"   GPUs: {args.num_gpus}")

    print(f"\n⚙️  PPO CONFIGURATION:")
    print(f"   Training batch: {args.train_batch_size:,} timesteps")
    print(f"   Minibatch size: {args.sgd_minibatch_size}")
    print(f"   SGD iterations: {args.num_sgd_iter}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Gamma: {args.gamma}")

    print(f"\n🎯 TRAINING TARGET:")
    print(f"   Total timesteps: {args.stop_timesteps:,}")
    print(f"   Checkpoint every: {args.checkpoint_freq} iterations")

    print("="*80 + "\n")

    # Initialize Ray
    print("🔧 Initializing Ray...")
    ray.init(ignore_reinit_error=True, include_dashboard=True)
    print(f"✅ Ray initialized - Dashboard: http://127.0.0.1:8265")
    print(f"   Available resources: {ray.available_resources()}\n")

    # Register environment
    register_env("StreetFighter-v0", env_creator)

    # Configure PPO
    config = (
        PPOConfig()
        .environment(
            env="StreetFighter-v0",
            env_config={"frame_stack": args.frame_stack},
        )
        .framework("torch")
        .training(
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.lambda_,
            clip_param=args.clip_param,
            entropy_coeff=args.entropy_coeff,
            vf_loss_coeff=1.0,
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            # CNN model for image observations
            model={
                "conv_filters": [
                    [32, [8, 8], 4],   # 32 filters, 8x8 kernel, stride 4
                    [64, [4, 4], 2],   # 64 filters, 4x4 kernel, stride 2
                    [64, [3, 3], 1],   # 64 filters, 3x3 kernel, stride 1
                ],
                "fcnet_hiddens": [512],
                "fcnet_activation": "relu",
            },
        )
        .rollouts(
            num_rollout_workers=args.num_workers,
            num_envs_per_worker=args.num_envs_per_worker,
            rollout_fragment_length=args.rollout_fragment_length,
        )
        .resources(
            num_gpus=args.num_gpus,
            num_cpus_per_worker=1,
        )
    )

    # Run training
    print("🚀 Starting training...\n")

    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"timesteps_total": args.stop_timesteps},
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        local_dir=os.path.expanduser(args.checkpoint_dir),
        restore=args.resume,
        verbose=1,
        callbacks=[WinRateCallback()],
    )

    print("\n" + "="*80)
    print("🏁 TRAINING COMPLETE!")
    print("="*80)

    # Get best checkpoint
    best_checkpoint = results.get_best_checkpoint(
        results.trials[0], mode="max", metric="episode_reward_mean"
    )
    print(f"\n✅ Best checkpoint: {best_checkpoint}")

    ray.shutdown()


if __name__ == "__main__":
    main()
