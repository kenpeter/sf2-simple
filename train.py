import os
import sys
import argparse
import time

import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import the wrapper
from wrapper import StreetFighterCustomWrapper


def make_env(game, state, seed=0, rendering=False):
    """Create environment with wrapper"""

    def _init():
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if rendering else None,
        )

        env = StreetFighterCustomWrapper(
            env,
            reset_round=True,
            rendering=rendering,
            max_episode_steps=5000,
        )

        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def linear_schedule(initial_value, final_value=0.0):
    """Linear scheduler for learning rate"""

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def main():
    parser = argparse.ArgumentParser(description="Train Street Fighter II Agent")
    parser.add_argument(
        "--total-timesteps", type=int, default=10000000, help="Total timesteps to train"
    )
    parser.add_argument(
        "--num-envs", type=int, default=16, help="Number of parallel environments"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from saved model path"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--use-original-state",
        action="store_true",
        help="Use original state file instead of absolute path",
    )
    parser.add_argument(
        "--list-states", action="store_true", help="List available states and exit"
    )

    args = parser.parse_args()

    # List available states if requested
    if args.list_states:
        try:
            import retro

            game = "StreetFighterIISpecialChampionEdition-Genesis"
            states = retro.data.list_states(game)
            print(f"📋 Available states for {game}:")
            for state in states:
                print(f"   - {state}")
        except Exception as e:
            print(f"❌ Could not list states: {e}")
        return

    game = "StreetFighterIISpecialChampionEdition-Genesis"

    # Handle state file path properly
    if args.use_original_state:
        state_file = "ken_bison_12.state"
        print("🎮 Using original state file: ken_bison_12.state")
    else:
        state_file = os.path.abspath("ken_bison_12.state")
        if not os.path.exists("ken_bison_12.state"):
            print(f"❌ State file not found: ken_bison_12.state")
            print("🔍 Current directory files:")
            for f in os.listdir("."):
                if f.endswith(".state"):
                    print(f"   - {f}")
            print(
                "💡 Try using --use-original-state flag or --list-states to see available states"
            )
            return
        print(f"🎮 Using absolute state path: {state_file}")

    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    print(f"🚀 Street Fighter II Training")
    print(f"   Total timesteps: {args.total_timesteps:,}")
    print(f"   Environments: {args.num_envs}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   State file: {state_file}")
    if args.resume:
        print(f"   Resuming from: {args.resume}")

    # Create environments
    print(f"🔧 Creating {args.num_envs} parallel environments...")
    try:
        env = SubprocVecEnv(
            [
                make_env(game, state=state_file, seed=i, rendering=args.render)
                for i in range(args.num_envs)
            ]
        )
        print("✅ Environments created successfully")
    except Exception as e:
        print(f"❌ Failed to create environments: {e}")
        print("💡 Try using --use-original-state flag or check state file path")
        return

    # Create or load model
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading model from: {args.resume}")
        model = PPO.load(args.resume, env=env, device="cuda")

        # Update learning rate for resumed training
        lr_schedule = linear_schedule(args.learning_rate, args.learning_rate * 0.1)
        model.learning_rate = lr_schedule
        print("✅ Model loaded, resuming training")

    else:
        print("🧠 Creating new PPO model")
        lr_schedule = linear_schedule(args.learning_rate, args.learning_rate * 0.1)

        model = PPO(
            "CnnPolicy",
            env,
            device="cuda",
            verbose=1,
            n_steps=1024,
            batch_size=256,
            n_epochs=8,
            gamma=0.995,
            learning_rate=lr_schedule,
            clip_range=linear_schedule(0.2, 0.05),
            ent_coef=0.01,
            vf_coef=0.8,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            tensorboard_log="logs",
        )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000000 // args.num_envs,
        save_path=save_dir,
        name_prefix="ppo_sf2",
    )

    # Training
    start_time = time.time()
    print(f"🏋️ Starting training for {args.total_timesteps:,} timesteps")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback],
            reset_num_timesteps=not bool(args.resume),  # Don't reset if resuming
        )

        training_time = time.time() - start_time
        print(f"🎉 Training completed in {training_time/3600:.1f} hours!")

    except KeyboardInterrupt:
        print(f"⏹️ Training interrupted")
        training_time = time.time() - start_time
        print(f"Training time: {training_time/3600:.1f} hours")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    finally:
        env.close()

    # Always save final model
    final_model_path = os.path.join(save_dir, "ppo_sf2_final.zip")
    model.save(final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")

    print("✅ Training complete!")
    print(f"🎮 Test with: python eval.py --model-path {final_model_path}")
    print(
        f"🔄 Resume with: python train.py --resume {final_model_path} --learning-rate {args.learning_rate}"
    )


if __name__ == "__main__":
    main()
