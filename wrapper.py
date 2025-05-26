import math
import time
import collections

import gymnasium as gym
import numpy as np


# Custom environment wrapper - matching original implementation
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # Frame stacking configuration - ORIGINAL VALUES
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)
        self.num_step_frames = 6

        # Reward system parameters - ORIGINAL VALUES (no normalization!)
        self.reward_coeff = 3  # INTEGER like original, not float
        self.full_hp = 176

        # Health tracking
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0

        # Observation space configuration
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(100, 128, 3), dtype=np.uint8
        )

        self.reset_round = reset_round
        self.rendering = rendering

        print(f"StreetFighterCustomWrapper initialized (ORIGINAL IMPLEMENTATION):")
        print(f"- Reward coefficient: {self.reward_coeff}")
        print(f"- Full HP: {self.full_hp}")
        print(f"- NO normalization factor (like original)")

    def _stack_observation(self):
        """Stack frames for observation"""
        return np.stack(
            [self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1
        )

    def _calculate_reward(self, curr_player_health, curr_oppont_health):
        """
        Simple damage-based reward: +1 for damage dealt, -1 for damage taken
        """
        custom_reward = 0.0
        custom_done = False

        # Check if game is over
        if curr_player_health <= 0 or curr_oppont_health <= 0:
            custom_done = True

        # Calculate damage-based rewards
        damage_dealt = self.prev_oppont_health - curr_oppont_health
        damage_received = self.prev_player_health - curr_player_health

        # +1 for each point of damage dealt, -1 for each point of damage received
        custom_reward = damage_dealt - damage_received

        # Update health tracking for next step
        self.prev_player_health = curr_player_health
        self.prev_oppont_health = curr_oppont_health

        # When reset_round flag is set to False, never reset episodes
        if not self.reset_round:
            custom_done = False

        return custom_reward, custom_done

    def reset(self, **kwargs):
        """Reset environment"""
        # Updated for gymnasium compatibility
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

        # Reset health tracking
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp
        self.total_timesteps = 0

        # Clear and initialize frame stack
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])

        stacked_obs = self._stack_observation()
        return stacked_obs, info

    def step(self, action):
        """Step function with ORIGINAL reward calculation"""
        # Handle MultiBinary action space
        if isinstance(self.env.action_space, gym.spaces.MultiBinary):
            if isinstance(action, np.ndarray):
                action = action.astype(int)
            elif isinstance(action, (list, tuple)):
                action = np.array(action, dtype=int)
            else:
                # Convert single integer to MultiBinary format
                binary_action = np.zeros(self.env.action_space.n, dtype=int)
                if 0 <= action < self.env.action_space.n:
                    binary_action[action] = 1
                action = binary_action

        # First step
        result = self.env.step(action)
        if len(result) == 5:
            obs, _reward, terminated, truncated, info = result
            _done = terminated or truncated
        else:
            obs, _reward, _done, info = result
            terminated = _done
            truncated = False

        self.frame_stack.append(obs[::2, ::2, :])

        # Render if enabled
        if self.rendering:
            self.env.render()
            time.sleep(0.01)

        # Execute action for additional frames (num_step_frames - 1)
        for _ in range(self.num_step_frames - 1):
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

        # Extract current health values from info
        curr_player_health = info.get("agent_hp", self.full_hp)
        curr_oppont_health = info.get("enemy_hp", self.full_hp)

        self.total_timesteps += self.num_step_frames

        # Calculate ORIGINAL reward (no normalization)
        custom_reward, custom_done = self._calculate_reward(
            curr_player_health, curr_oppont_health
        )

        # Return in gymnasium-compatible format
        if len(result) == 5:
            return (
                self._stack_observation(),
                custom_reward,  # Raw reward, no normalization
                custom_done,
                False,  # truncated
                info,
            )
        else:
            return self._stack_observation(), custom_reward, custom_done, info
