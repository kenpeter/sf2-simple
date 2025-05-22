import math
import time
import collections

import gymnasium as gym
import numpy as np


# Custom environment wrapper - fixed for stable-retro
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

        # Disable auto-start for now
        self.auto_start = False

        # Track if we're using info dict for HP (fallback)
        self.use_info_hp = True

        # These memory addresses might not work for your ROM version
        # We'll rely on the info dict instead
        self.game_info = {
            "enemy_character": {"address": 16745563, "type": "|u1"},
            "agent_hp": {"address": 16744514, "type": ">i2"},
            "agent_x": {"address": 16744454, "type": ">u2"},
            "agent_y": {"address": 16744458, "type": ">u2"},
            "enemy_hp": {"address": 16745154, "type": ">i2"},
            "enemy_x": {"address": 16745094, "type": ">u2"},
            "enemy_y": {"address": 16745098, "type": ">u2"},
            "score": {"address": 16744936, "type": ">d4"},
            "agent_victories": {"address": 16744922, "type": "|u1"},
            "enemy_victories": {"address": 16745559, "type": ">u4"},
            "round_countdown": {"address": 16750378, "type": ">u2"},
            "reset_countdown": {"address": 16744917, "type": "|u1"},
            "agent_status": {"address": 16744450, "type": ">u2"},
            "enemy_status": {"address": 16745090, "type": ">u2"},
        }

    def _stack_observation(self):
        return np.stack(
            [self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1
        )

    def _get_game_info(self):
        """Extract game information - use info dict as primary source"""
        game_state = {
            "agent_hp": self.full_hp,
            "enemy_hp": self.full_hp,
            "agent_x": 0,
            "agent_y": 0,
            "enemy_x": 0,
            "enemy_y": 0,
            "score": 0,
            "agent_victories": 0,
            "enemy_victories": 0,
            "round_countdown": 0,
            "reset_countdown": 0,
            "agent_status": 0,
            "enemy_status": 0,
            "enemy_character": 0,
        }

        # Try to get RAM data, but don't rely on it
        try:
            ram = self.env.get_ram()
            if len(ram) > 1000:  # Basic sanity check
                # Try a few different potential HP addresses
                potential_hp_addresses = [
                    (0x1000, 0x1100),  # Common area 1
                    (0x0400, 0x0500),  # Common area 2
                    (0xFF0000, 0xFF1000),  # Your original addresses
                ]

                for start_addr, end_addr in potential_hp_addresses:
                    for addr in range(start_addr, min(end_addr, len(ram) - 1), 2):
                        try:
                            hp_val = int.from_bytes(
                                ram[addr : addr + 2], byteorder="big", signed=True
                            )
                            if 0 <= hp_val <= 200:  # Reasonable HP range
                                game_state["agent_hp"] = hp_val
                                # Look for enemy HP nearby
                                for offset in [100, 200, 300, 640]:  # Common offsets
                                    if addr + offset < len(ram) - 1:
                                        enemy_hp = int.from_bytes(
                                            ram[addr + offset : addr + offset + 2],
                                            byteorder="big",
                                            signed=True,
                                        )
                                        if 0 <= enemy_hp <= 200:
                                            game_state["enemy_hp"] = enemy_hp
                                            return game_state
                        except:
                            continue
        except:
            pass

        return game_state

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

        # Get game state info and update info dict
        game_state = self._get_game_info()

        # Use info dict HP values if available and reasonable
        if "hp" in info or "health" in info or "agent_hp" in info:
            for key in info:
                if "hp" in key.lower() or "health" in key.lower():
                    hp_val = info[key]
                    if isinstance(hp_val, (int, float)) and 0 <= hp_val <= 200:
                        if "agent" in key.lower() or "player" in key.lower():
                            game_state["agent_hp"] = int(hp_val)
                        elif "enemy" in key.lower() or "opponent" in key.lower():
                            game_state["enemy_hp"] = int(hp_val)

        info.update(game_state)

        # Print debug info only once
        if not hasattr(self, "_debug_printed"):
            print(f"Action space: {self.env.action_space}")
            print(f"Available info keys: {list(info.keys())}")
            print(
                f"Game state HP: agent={game_state.get('agent_hp')}, enemy={game_state.get('enemy_hp')}"
            )
            self._debug_printed = True

        stacked_obs = np.stack(
            [self.frame_stack[i * 3 + 2][:, :, i] for i in range(3)], axis=-1
        )
        return stacked_obs, info

    def step(self, action):
        custom_done = False

        # Handle MultiBinary action space - convert to proper format
        if isinstance(self.env.action_space, gym.spaces.MultiBinary):
            # action should be an array of 0s and 1s for each button
            if isinstance(action, np.ndarray):
                action = action.astype(int)
            elif isinstance(action, (list, tuple)):
                action = np.array(action, dtype=int)
            else:
                # If we get a single integer, convert to MultiBinary format
                # This shouldn't happen with PPO, but just in case
                binary_action = np.zeros(self.env.action_space.n, dtype=int)
                if 0 <= action < self.env.action_space.n:
                    binary_action[action] = 1
                action = binary_action

        # Updated for gymnasium compatibility
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

        # Get game state info and update info dict
        game_state = self._get_game_info()

        # Try to extract HP from info dict first
        curr_player_health = self.full_hp
        curr_oppont_health = self.full_hp

        # Look for HP in info dict
        for key, value in info.items():
            if isinstance(value, (int, float)) and 0 <= value <= 200:
                if "agent" in key.lower() or "player" in key.lower():
                    if "hp" in key.lower() or "health" in key.lower():
                        curr_player_health = int(value)
                elif "enemy" in key.lower() or "opponent" in key.lower():
                    if "hp" in key.lower() or "health" in key.lower():
                        curr_oppont_health = int(value)

        # Fallback to game_state if info dict doesn't have HP
        if (
            curr_player_health == self.full_hp
            and game_state.get("agent_hp", self.full_hp) != self.full_hp
        ):
            curr_player_health = game_state.get("agent_hp", self.full_hp)
        if (
            curr_oppont_health == self.full_hp
            and game_state.get("enemy_hp", self.full_hp) != self.full_hp
        ):
            curr_oppont_health = game_state.get("enemy_hp", self.full_hp)

        info.update(game_state)
        self.total_timesteps += self.num_step_frames

        # Simple reward function - if we can't get HP, just use default rewards
        if curr_player_health == self.full_hp and curr_oppont_health == self.full_hp:
            # No HP data available, use time-based or action-based rewards
            custom_reward = 0.1  # Small positive reward for survival
        else:
            # Game is over and player loses.
            if curr_player_health <= 0:
                custom_reward = -math.pow(
                    self.full_hp, (curr_oppont_health + 1) / (self.full_hp + 1)
                )
                custom_done = True

            # Game is over and player wins.
            elif curr_oppont_health <= 0:
                custom_reward = (
                    math.pow(
                        self.full_hp, (curr_player_health + 1) / (self.full_hp + 1)
                    )
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
