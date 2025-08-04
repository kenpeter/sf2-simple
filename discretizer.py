"""
Define discrete action spaces for Gym Retro environments with a limited set of button combos
All credit to open-ai's examples: https://github.com/openai/retro/blob/master/retro/examples/discretizer.py
"""

import gymnasium as gym
import numpy as np


# discretizer
# param gym action wrapper
class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    # self, env, combo
    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        self._combos = combos

        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()

    def get_action_meaning(self, act):
        return self._combos[act]


class StreetFighter2Discretizer(Discretizer):
    """
    Use Street Fighter 2 special moves
    A, B, C = punches (light, medium, heavy)
    X, Y, Z = kicks (light, medium, heavy)
    """

    def __init__(self, env):
        super().__init__(
            env=env,
            combos=[
                [],  # No input
                ["UP"],
                ["DOWN"],
                ["LEFT"],
                ["UP", "LEFT"],
                ["DOWN", "LEFT"],
                ["RIGHT"],
                ["UP", "RIGHT"],
                ["DOWN", "RIGHT"],
                ["B"],  # Light punch
                ["B", "DOWN"],
                ["B", "LEFT"],
                ["B", "RIGHT"],
                ["A"],  # Medium punch
                ["A", "DOWN"],
                ["A", "LEFT"],
                ["A", "RIGHT"],
                ["C"],  # Heavy punch
                ["DOWN", "C"],
                ["LEFT", "C"],
                ["RIGHT", "C"],
                ["Y"],  # Light kick
                ["DOWN", "Y"],
                ["LEFT", "Y"],
                ["DOWN", "LEFT", "Y"],
                ["RIGHT", "Y"],
                ["X"],  # Medium kick
                ["DOWN", "X"],
                ["LEFT", "X"],
                ["DOWN", "LEFT", "X"],
                ["RIGHT", "X"],
                ["DOWN", "RIGHT", "X"],
                ["Z"],  # Heavy kick
                ["DOWN", "Z"],
                ["LEFT", "Z"],
                ["DOWN", "LEFT", "Z"],
                ["RIGHT", "Z"],
                ["DOWN", "RIGHT", "Z"],
                # Special move combinations for Ken/Ryu (facing right)
                ["DOWN", "DOWN", "RIGHT", "B"],  # Hadoken (facing right)
                ["RIGHT", "DOWN", "DOWN", "RIGHT", "B"],  # Dragon punch (facing right)
                ["LEFT", "DOWN", "LEFT", "Y"],  # Hurricane kick (facing right)
                # Special move combinations for Ken/Ryu (facing left)
                ["DOWN", "DOWN", "LEFT", "B"],  # Hadoken (facing left)
                ["LEFT", "DOWN", "DOWN", "LEFT", "B"],  # Dragon punch (facing left)
                ["RIGHT", "DOWN", "RIGHT", "Y"],  # Hurricane kick (facing left)
            ],
        )
