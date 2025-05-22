import math
import time
import collections

import gymnasium as gym
import numpy as np


# Custom environment wrapper
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # Use a deque to store the last 9 frames
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        self.num_step_frames = 6

        self.reward_coeff = 3.0

        self.total_timesteps = 0

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(100, 128, 3), dtype=np.uint8
        )

        self.reset_round = reset_round
        self.rendering = rendering

    def _stack_observation(self):
        return np.stack(
            [self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1
        )

    def reset(self, **kwargs):
        # Updated for gymnasium compatibility - reset returns (obs, info)
        if hasattr(self.env, "reset"):
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple):
                observation, info = result
            else:
                observation = result
                info = {}
        else:
            observation = self.env.reset()
            info = {}

        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0

        # Clear the frame stack and add the first observation [num_frames] times
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])

        stacked_obs = np.stack(
            [self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1
        )
        return stacked_obs, info

    def step(self, action):
        custom_done = False

        # Updated for gymnasium compatibility - step returns (obs, reward, terminated, truncated, info)
        result = self.env.step(action)
        if len(result) == 5:
            obs, _reward, terminated, truncated, info = result
            _done = terminated or truncated
        else:
            obs, _reward, _done, info = result
            terminated = _done
            truncated = False

        self.frame_stack.append(obs[::2, ::2, :])

        # Render the game if rendering flag is set to True.
        if self.rendering:
            self.env.render()
            time.sleep(0.01)

        for _ in range(self.num_step_frames - 1):

            # Keep the button pressed for (num_step_frames - 1) frames.
            result = self.env.step(action)
            if len(result) == 5:
                obs, _reward, terminated, truncated, info = result
                _done = terminated or truncated
            else:
                obs, _reward, _done, info = result

            self.frame_stack.append(obs[::2, ::2, :])
            if self.rendering:
                self.env.render()
                time.sleep(0.01)

        curr_player_health = info["agent_hp"]
        curr_oppont_health = info["enemy_hp"]

        self.total_timesteps += self.num_step_frames

        # Game is over and player loses.
        if curr_player_health < 0:
            custom_reward = -math.pow(
                self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1)
            )  # Use the remaining health points of opponent as penalty.
            # If the opponent also has negative health points, it's a even game and the reward is +1.
            custom_done = True

        # Game is over and player wins.
        elif curr_oppont_health < 0:
            # custom_reward = curr_player_health * self.reward_coeff # Use the remaining health points of player as reward.
            # Multiply by reward_coeff to make the reward larger than the penalty to avoid cowardice of agent.

            # custom_reward = math.pow(self.full_hp, (5940 - self.total_timesteps) / 5940) * self.reward_coeff # Use the remaining time steps as reward.
            custom_reward = (
                math.pow(self.full_hp, (curr_player_health + 1) / (self.full_hp + 1))
                * self.reward_coeff
            )
            custom_done = True

        # While the fighting is still going on
        else:
            custom_reward = self.reward_coeff * (
                self.prev_oppont_health - curr_oppont_health
            ) - (self.prev_player_health - curr_player_health)
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            custom_done = False

        # When reset_round flag is set to False (never reset), the session should always keep going.
        if not self.reset_round:
            custom_done = False

        # Max reward is 6 * full_hp = 1054 (damage * 3 + winning_reward * 3) norm_coefficient = 0.001
        normalized_reward = 0.001 * custom_reward

        # Return gymnasium-compatible format
        if len(result) == 5:
            return (
                self._stack_observation(),
                normalized_reward,
                custom_done,
                False,
                info,
            )
        else:
            return self._stack_observation(), normalized_reward, custom_done, info
