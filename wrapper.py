import math
import time
import collections

import gymnasium as gym
import numpy as np


# Custom environment wrapper - optimized with HP address caching and aligned rewards
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(
        self, env, reset_round=True, rendering=False, reward_coeff=3.0, full_hp=176
    ):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # Frame stacking configuration
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)
        self.num_step_frames = 6

        # Reward system parameters - now configurable and aligned
        self.reward_coeff = reward_coeff
        self.full_hp = full_hp
        self.reward_normalization = 0.001  # Consistent normalization factor

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

        # HP detection - simplified approach matching reference
        self.use_info_hp = True

        # Debug info
        self._debug_printed = False

        print(f"StreetFighterCustomWrapper initialized:")
        print(f"- Reward coefficient: {self.reward_coeff}")
        print(f"- Full HP: {self.full_hp}")
        print(f"- Normalization factor: {self.reward_normalization}")
        print(
            f"- Max reward per step: {self.reward_coeff * self.full_hp * self.reward_normalization:.3f}"
        )

    def _get_hp_from_info(self, info):
        """Extract HP values from info dict - simplified approach matching reference"""
        agent_hp = self.full_hp
        enemy_hp = self.full_hp

        # Direct extraction from info dict (matching reference implementation)
        if "agent_hp" in info:
            agent_hp = info["agent_hp"]
        if "enemy_hp" in info:
            enemy_hp = info["enemy_hp"]

        return agent_hp, enemy_hp

    def _stack_observation(self):
        """Stack frames for observation"""
        return np.stack(
            [self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1
        )

    def _calculate_reward(self, curr_player_health, curr_oppont_health):
        """
        Aligned reward calculation matching the reference implementation

        Reward structure:
        - Win: exponential reward based on remaining HP * reward_coeff
        - Loss: negative exponential penalty based on opponent's remaining HP
        - Ongoing: reward_coeff * damage_dealt - damage_taken
        - All rewards normalized by 0.001
        """
        custom_reward = 0.0
        custom_done = False

        # Game is over and player loses (HP <= 0, not < 0 to match reference)
        if curr_player_health <= 0:
            # Exponential penalty based on opponent's remaining health
            # If opponent also has <= 0 HP, it's a draw and penalty is minimal
            custom_reward = -math.pow(
                self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1)
            )
            custom_done = True

        # Game is over and player wins (opponent HP <= 0)
        elif curr_oppont_health <= 0:
            # Exponential reward based on player's remaining health * reward coefficient
            custom_reward = (
                math.pow(self.full_hp, (curr_player_health + 1) / (self.full_hp + 1))
                * self.reward_coeff
            )
            custom_done = True

        # While the fighting is still going on
        else:
            # Reward for damage dealt to opponent, penalty for damage received
            damage_dealt = self.prev_oppont_health - curr_oppont_health
            damage_received = self.prev_player_health - curr_player_health

            custom_reward = (self.reward_coeff * damage_dealt) - damage_received

            # Update health tracking for next step
            self.prev_player_health = curr_player_health
            self.prev_oppont_health = curr_oppont_health
            custom_done = False

        # When reset_round flag is set to False, never reset episodes
        if not self.reset_round:
            custom_done = False

        # Apply consistent normalization
        # Max theoretical reward is reward_coeff * full_hp (for max damage or win)
        normalized_reward = self.reward_normalization * custom_reward

        return normalized_reward, custom_done

    def reset(self, **kwargs):
        """Reset environment with aligned initialization"""
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

        # Debug info (print once)
        if not self._debug_printed:
            print(f"Action space: {self.env.action_space}")
            print(f"Available info keys: {list(info.keys())}")
            self._debug_printed = True

        stacked_obs = self._stack_observation()
        return stacked_obs, info

    def step(self, action):
        """Step function with aligned reward calculation"""
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

        # Extract current health values directly from info (matching reference)
        curr_player_health = info.get("agent_hp", self.full_hp)
        curr_oppont_health = info.get("enemy_hp", self.full_hp)

        self.total_timesteps += self.num_step_frames

        # Calculate aligned reward
        normalized_reward, custom_done = self._calculate_reward(
            curr_player_health, curr_oppont_health
        )

        # Return in gymnasium-compatible format
        if len(result) == 5:
            return (
                self._stack_observation(),
                normalized_reward,
                custom_done,
                False,  # truncated
                info,
            )
        else:
            return self._stack_observation(), normalized_reward, custom_done, info
