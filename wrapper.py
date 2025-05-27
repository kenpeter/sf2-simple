import time
import collections

import gymnasium as gym
import numpy as np


# Custom environment wrapper - SIMPLE episode length limiting to avoid victory screens
class StreetFighterCustomWrapper(gym.Wrapper):
    def __init__(self, env, reset_round=True, rendering=False, max_episode_steps=2000):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # Action filtering to prevent cheating (built-in to avoid pickling issues)
        # Street Fighter II MultiBinary actions:
        # [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R]
        # We want to disable START (index 3) and SELECT (index 2)
        self.disabled_buttons = [2, 3]  # SELECT and START

        # Frame stacking configuration - ORIGINAL VALUES
        self.num_frames = 9
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        # Reward system parameters - ORIGINAL VALUES (no normalization!)
        self.reward_coeff = 3  # INTEGER like original, not float
        self.full_hp = 176

        # Health tracking
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        self.total_timesteps = 0

        # Win/Loss tracking
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0
        self.current_round_active = True

        # Simple episode length limiting to avoid victory screens
        self.max_episode_steps = max_episode_steps  # Reset after this many steps
        self.episode_steps = 0

        # Win rate trending - Simple approach
        self.win_rate_history = []
        self.trend_window_size = 20  # Number of data points to keep for trending
        self.rounds_per_trend_update = 10  # Update trend every N rounds
        self.last_trend_update_round = 0

        # Observation space configuration
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(100, 128, 3), dtype=np.uint8
        )

        self.reset_round = reset_round
        self.rendering = rendering

        print(f"StreetFighterCustomWrapper initialized:")
        print(f"- Reward coefficient: {self.reward_coeff}")
        print(f"- Full HP: {self.full_hp}")
        print(
            f"- Max episode steps: {self.max_episode_steps} (prevents victory screen stuck)"
        )
        print(f"- Win rate tracking enabled")
        print(
            f"- Win rate trending enabled (window: {self.trend_window_size}, update every {self.rounds_per_trend_update} rounds)"
        )
        print(f"- Action filtering enabled (SELECT and START buttons disabled)")

    def _stack_observation(self):
        """Stack frames for observation"""
        return np.stack(
            [self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1
        )

    def _update_win_rate_trend(self):
        """Update win rate trend history"""
        if (
            self.total_rounds
            >= self.last_trend_update_round + self.rounds_per_trend_update
        ):
            current_win_rate = self.get_win_rate()
            timestamp = time.time()

            self.win_rate_history.append(
                {
                    "round": self.total_rounds,
                    "win_rate": current_win_rate,
                    "timestamp": timestamp,
                }
            )

            # Keep only the most recent entries
            if len(self.win_rate_history) > self.trend_window_size:
                self.win_rate_history.pop(0)

            self.last_trend_update_round = self.total_rounds

    def get_win_rate_trend(self):
        """Calculate win rate trend - simple linear trend"""
        if len(self.win_rate_history) < 3:
            return "insufficient_data"

        # Get recent win rates
        recent_rates = [entry["win_rate"] for entry in self.win_rate_history[-5:]]

        # Simple trend calculation: compare first half vs second half
        mid_point = len(recent_rates) // 2
        first_half_avg = (
            np.mean(recent_rates[:mid_point]) if mid_point > 0 else recent_rates[0]
        )
        second_half_avg = np.mean(recent_rates[mid_point:])

        trend_threshold = 0.05  # 5% change to be considered significant

        if second_half_avg > first_half_avg + trend_threshold:
            return "improving"
        elif second_half_avg < first_half_avg - trend_threshold:
            return "declining"
        else:
            return "stable"

    def get_trend_stats(self):
        """Get detailed trend statistics"""
        if len(self.win_rate_history) == 0:
            return {
                "trend": "no_data",
                "current_win_rate": self.get_win_rate(),
                "trend_points": 0,
                "recent_avg": 0.0,
                "peak_win_rate": 0.0,
                "lowest_win_rate": 0.0,
            }

        win_rates = [entry["win_rate"] for entry in self.win_rate_history]

        return {
            "trend": self.get_win_rate_trend(),
            "current_win_rate": self.get_win_rate(),
            "trend_points": len(self.win_rate_history),
            "recent_avg": (
                np.mean(win_rates[-5:]) if len(win_rates) >= 5 else np.mean(win_rates)
            ),
            "peak_win_rate": max(win_rates),
            "lowest_win_rate": min(win_rates),
            "trend_history": self.win_rate_history[-10:],  # Last 10 data points
        }

    def _calculate_reward(self, curr_player_health, curr_oppont_health):
        """
        Simple damage-based reward: +1 for damage dealt, -1 for damage taken
        Now also tracks wins/losses and updates trending
        """
        custom_reward = 0.0
        custom_done = False

        # Simple episode length limit to avoid victory screens
        if self.episode_steps >= self.max_episode_steps:
            print(
                f"Episode length limit reached ({self.max_episode_steps} steps). Resetting to avoid victory screens."
            )
            custom_done = True
            return custom_reward, custom_done

        # Check if round is over and track win/loss
        round_over = False
        if curr_player_health <= 0 or curr_oppont_health <= 0:
            if self.current_round_active:  # Only count once per round
                self.total_rounds += 1
                if curr_oppont_health <= 0:
                    self.wins += 1
                    custom_reward += 100  # Bonus for winning
                    print(
                        f"VICTORY! Win #{self.wins} (Win rate: {self.get_win_rate():.1%})"
                    )
                elif curr_player_health <= 0:
                    self.losses += 1
                    custom_reward -= 50  # Penalty for losing
                    print(
                        f"DEFEAT! Loss #{self.losses} (Win rate: {self.get_win_rate():.1%})"
                    )

                self.current_round_active = False
                round_over = True

                # Update trend tracking
                self._update_win_rate_trend()

            # Reset after a short delay to allow some victory animation
            if (
                self.episode_steps > self.max_episode_steps - 100
            ):  # Reset near end anyway
                custom_done = True

        # Calculate damage-based rewards
        damage_dealt = self.prev_oppont_health - curr_oppont_health
        damage_received = self.prev_player_health - curr_player_health

        # +1 for each point of damage dealt, -1 for each point of damage received
        damage_reward = damage_dealt - damage_received
        custom_reward += damage_reward

        # Update health tracking for next step
        self.prev_player_health = curr_player_health
        self.prev_oppont_health = curr_oppont_health

        # When reset_round flag is set to False, never reset episodes
        if not self.reset_round:
            custom_done = False

        return custom_reward, custom_done

    def get_win_rate(self):
        """Calculate current win rate"""
        if self.total_rounds == 0:
            return 0.0
        return self.wins / self.total_rounds

    def get_win_stats(self):
        """Get detailed win statistics including trending"""
        base_stats = {
            "wins": self.wins,
            "losses": self.losses,
            "total_rounds": self.total_rounds,
            "win_rate": self.get_win_rate(),
            "episode_steps": self.episode_steps,
            "max_episode_steps": self.max_episode_steps,
        }

        # Add trending information
        trend_stats = self.get_trend_stats()
        base_stats.update(trend_stats)

        return base_stats

    def print_win_rate_summary(self):
        """Print a simple summary of win rate and trend"""
        stats = self.get_win_stats()
        print(f"\n=== Win Rate Summary ===")
        print(
            f"Current: {stats['current_win_rate']:.1%} ({stats['wins']}/{stats['total_rounds']})"
        )
        print(f"Trend: {stats['trend'].replace('_', ' ').title()}")
        if stats["trend_points"] > 0:
            print(f"Recent Avg: {stats['recent_avg']:.1%}")
            print(f"Peak: {stats['peak_win_rate']:.1%}")
        print(f"Episode: {stats['episode_steps']}/{stats['max_episode_steps']} steps")
        print(f"========================")

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
        self.current_round_active = True

        # Reset episode step counter
        self.episode_steps = 0

        # Clear and initialize frame stack
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(observation[::2, ::2, :])

        stacked_obs = self._stack_observation()

        # Add win rate info to the info dict
        info.update(self.get_win_stats())

        return stacked_obs, info

    def step(self, action):
        """Step function with ORIGINAL reward calculation + win tracking + trending + simple episode limiting"""
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

        # Apply action filtering to prevent cheating (disable SELECT and START)
        if hasattr(action, "__len__") and len(action) >= 4:
            filtered_action = action.copy() if hasattr(action, "copy") else list(action)
            for button_idx in self.disabled_buttons:
                if button_idx < len(filtered_action):
                    filtered_action[button_idx] = 0
            action = filtered_action

        # Execute the action
        observation, reward, done, truncated, info = self.env.step(action)

        # Get health information for reward calculation
        curr_player_health = info.get("health", self.full_hp)
        curr_oppont_health = info.get("enemy_health", self.full_hp)

        # Calculate custom reward and done condition
        custom_reward, custom_done = self._calculate_reward(
            curr_player_health, curr_oppont_health
        )

        # Override done if custom logic determines it
        if custom_done:
            done = custom_done

        # Update frame stack
        self.frame_stack.append(observation[::2, ::2, :])
        stacked_obs = self._stack_observation()

        # Update counters
        self.episode_steps += 1

        # Update info with win stats (including trending)
        info.update(self.get_win_stats())

        self.total_timesteps += 1

        return stacked_obs, custom_reward, done, truncated, info
