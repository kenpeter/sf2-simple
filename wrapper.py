import math
import time
import collections

import gymnasium as gym
import numpy as np


# Custom environment wrapper - optimized with HP address caching
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

        # HP address caching - OPTIMIZATION
        self.hp_addresses_found = False
        self.agent_hp_addr = None
        self.enemy_hp_addr = None
        self.hp_search_attempts = 0
        self.max_hp_search_attempts = 10  # Stop searching after 10 attempts

        # Disable auto-start for now
        self.auto_start = False

        # Track if we're using info dict for HP (fallback)
        self.use_info_hp = True

    def _find_hp_addresses(self):
        """Find HP addresses once and cache them - OPTIMIZED"""
        if (
            self.hp_addresses_found
            or self.hp_search_attempts >= self.max_hp_search_attempts
        ):
            return

        self.hp_search_attempts += 1

        try:
            ram = self.env.get_ram()
            if len(ram) < 1000:
                return

            # Try a few different potential HP addresses
            potential_hp_addresses = [
                (0x1000, 0x1100),  # Common area 1
                (0x0400, 0x0500),  # Common area 2
                (0x2000, 0x2100),  # Common area 3
            ]

            for start_addr, end_addr in potential_hp_addresses:
                for addr in range(start_addr, min(end_addr, len(ram) - 1), 2):
                    try:
                        hp_val = int.from_bytes(
                            ram[addr : addr + 2], byteorder="big", signed=True
                        )
                        if 0 <= hp_val <= 200:  # Reasonable HP range
                            # Look for enemy HP nearby
                            for offset in [100, 200, 300, 640]:  # Common offsets
                                if addr + offset < len(ram) - 1:
                                    enemy_hp = int.from_bytes(
                                        ram[addr + offset : addr + offset + 2],
                                        byteorder="big",
                                        signed=True,
                                    )
                                    if 0 <= enemy_hp <= 200:
                                        # Found valid HP addresses!
                                        self.agent_hp_addr = addr
                                        self.enemy_hp_addr = addr + offset
                                        self.hp_addresses_found = True
                                        print(
                                            f"HP addresses found: Agent={hex(addr)}, Enemy={hex(addr + offset)}"
                                        )
                                        return
                    except:
                        continue
        except:
            pass

    def _get_game_info(self):
        """Extract game information - OPTIMIZED with caching"""
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

        # If we haven't found HP addresses yet, try to find them
        if not self.hp_addresses_found:
            self._find_hp_addresses()

        # Use cached addresses if available
        if self.hp_addresses_found:
            try:
                ram = self.env.get_ram()
                if len(ram) > max(self.agent_hp_addr + 1, self.enemy_hp_addr + 1):
                    # Read agent HP
                    agent_hp = int.from_bytes(
                        ram[self.agent_hp_addr : self.agent_hp_addr + 2],
                        byteorder="big",
                        signed=True,
                    )
                    if 0 <= agent_hp <= 200:
                        game_state["agent_hp"] = agent_hp

                    # Read enemy HP
                    enemy_hp = int.from_bytes(
                        ram[self.enemy_hp_addr : self.enemy_hp_addr + 2],
                        byteorder="big",
                        signed=True,
                    )
                    if 0 <= enemy_hp <= 200:
                        game_state["enemy_hp"] = enemy_hp
            except:
                # If cached addresses fail, mark as not found to retry
                self.hp_addresses_found = False

        return game_state

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
            print(f"HP addresses cached: {self.hp_addresses_found}")
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

        # Get game state info - NOW OPTIMIZED with caching
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
