import collections
import gymnasium as gym
import numpy as np


class StreetFighterCustomWrapper(gym.Wrapper):
    """Simplified Street Fighter wrapper with frame stacking and health tracking"""

    def __init__(self, env, reset_round=True, rendering=False, max_episode_steps=5000):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # Frame stacking - Updated to 6 frames
        self.num_frames = 6
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        # Health tracking
        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # Episode management
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # Win tracking for display
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0

        # Observation space: 6 frames * 3 channels = 18 channels
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(100, 128, 18), dtype=np.uint8
        )

        print(f"🚀 StreetFighter Wrapper initialized:")
        print(f"   Observation shape: {self.observation_space.shape}")
        print(f"   Max episode steps: {self.max_episode_steps}")
        print(f"   Frame stacking: {self.num_frames} frames")

    def _stack_observation(self):
        """Stack recent frames"""
        # Fill stack if not enough frames
        while len(self.frame_stack) < self.num_frames:
            if len(self.frame_stack) > 0:
                self.frame_stack.append(self.frame_stack[-1].copy())
            else:
                self.frame_stack.append(np.zeros((100, 128, 3), dtype=np.uint8))

        # Concatenate frames along channel dimension
        stacked = np.concatenate(list(self.frame_stack), axis=-1)
        return stacked

    def _extract_health(self, info):
        """Extract health from info"""
        player_health = info.get("agent_hp", self.full_hp)
        opponent_health = info.get("enemy_hp", self.full_hp)
        return player_health, opponent_health

    def _calculate_reward(self, curr_player_health, curr_opponent_health):
        """Calculate reward based on damage dealt/received"""
        reward = 0.0
        done = False

        # Episode timeout
        if self.episode_steps >= self.max_episode_steps:
            done = True
            return reward, done

        # Check for round end
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1

            if curr_opponent_health <= 0 and curr_player_health > 0:
                self.wins += 1
                win_rate = self.wins / self.total_rounds
                print(f"🏆 WIN! {self.wins}/{self.total_rounds} ({win_rate:.1%})")
            elif curr_player_health <= 0 and curr_opponent_health > 0:
                self.losses += 1
                win_rate = self.wins / self.total_rounds
                print(f"💀 LOSS! {self.wins}/{self.total_rounds} ({win_rate:.1%})")

            if self.reset_round:
                done = True

        # Damage-based reward: +1 per damage dealt, -1 per damage received
        damage_dealt = max(0, self.prev_opponent_health - curr_opponent_health)
        damage_received = max(0, self.prev_player_health - curr_player_health)
        reward = damage_dealt - damage_received

        # Update health tracking
        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        return reward, done

    def reset(self, **kwargs):
        """Reset environment"""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            observation, info = result
        else:
            observation = result
            info = {}

        # Reset tracking
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0

        # Initialize frame stack
        self.frame_stack.clear()
        downsampled = observation[::2, ::2, :]  # Downsample to (100, 128, 3)

        for _ in range(self.num_frames):
            self.frame_stack.append(downsampled.copy())

        stacked_obs = self._stack_observation()
        return stacked_obs, info

    def step(self, action):
        """Step function"""
        # Convert action if needed
        if isinstance(action, np.ndarray) and action.ndim == 0:
            # Convert scalar to binary action
            binary_action = np.zeros(self.env.action_space.n, dtype=int)
            if 0 <= action < self.env.action_space.n:
                binary_action[action] = 1
            action = binary_action

        # Execute step
        observation, reward, done, truncated, info = self.env.step(action)

        # Extract health and calculate custom reward
        curr_player_health, curr_opponent_health = self._extract_health(info)
        custom_reward, custom_done = self._calculate_reward(
            curr_player_health, curr_opponent_health
        )

        if custom_done:
            done = custom_done

        # Update frame stack
        downsampled = observation[::2, ::2, :]
        self.frame_stack.append(downsampled)
        stacked_obs = self._stack_observation()

        self.episode_steps += 1

        return stacked_obs, custom_reward, done, truncated, info
