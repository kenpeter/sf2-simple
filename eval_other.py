import retro
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage

from wrapper import StreetFighterCustomWrapper

RESET_ROUND = True  # Reset the round when fight is over.
RENDERING = False


def make_env(game, state):
    def _init():
        if state and os.path.exists(state):
            state_path = os.path.abspath(state)
        else:
            state_path = state

        env = retro.make(
            game=game,
            state=state_path,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
        )
        env = StreetFighterCustomWrapper(
            env, reset_round=RESET_ROUND, rendering=RENDERING
        )
        env = Monitor(env)
        return env

    return _init


game = "StreetFighterIISpecialChampionEdition-Genesis"
state_file = os.path.abspath("ken_bison_12.state")

# Create vectorized environment like in training
env = DummyVecEnv([make_env(game, state_file)])
env = VecTransposeImage(env)

# Load model correctly
model = PPO.load(
    "/home/kenpeter/work/sf2-simple/trained_models/ppo_ryu_7000000_steps_updated.zip",
    env=env,
)

mean_reward, std_reward = evaluate_policy(
    model,
    env,
    render=False,
    n_eval_episodes=5,
    deterministic=False,
    return_episode_rewards=True,
)
print(mean_reward)
print(std_reward)
# print(f"Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
