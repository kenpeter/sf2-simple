#!/usr/bin/env python3
"""
🥊 Street Fighter 2 Gameplay Script
Uses isolated QwenStreetFighterAgent for inference and gameplay
"""

import argparse
import retro
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
from qwen_agent import QwenStreetFighterAgent
from discretizer import StreetFighter2Discretizer

# Simple StreetFighter wrapper for gameplay
class StreetFighter(gym.Env):
    def __init__(self):
        super().__init__()
        game = retro.make(
            "StreetFighterIISpecialChampionEdition-Genesis",
            state="ken_bison_12.state",
            use_restricted_actions=retro.Actions.FILTERED,
        )
        self.game = StreetFighter2Discretizer(game)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = self.game.action_space
        self.agent_hp = 176
        self.enemy_hp = 176
    
    def preprocess(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resize, (84, 84, 1))
    
    def step(self, action):
        obs, _, done, truncated, info = self.game.step(action)
        obs = self.preprocess(obs)
        
        current_agent_hp = info.get("agent_hp", self.agent_hp)
        current_enemy_hp = info.get("enemy_hp", self.enemy_hp)
        
        delta_hp_diff = (current_agent_hp - self.agent_hp) - (current_enemy_hp - self.enemy_hp)
        reward = delta_hp_diff / 176.0 - 0.0001
        
        self.agent_hp = current_agent_hp
        self.enemy_hp = current_enemy_hp
        
        if self.agent_hp <= 0 or self.enemy_hp <= 0:
            done = True
            if self.enemy_hp <= 0 and self.agent_hp > 0:
                reward += 1.0
                info["agent_won"] = True
            else:
                reward -= 1.0
                info["agent_won"] = False
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        result = self.game.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        obs = self.preprocess(obs)
        self.agent_hp = info.get("agent_hp", 176)
        self.enemy_hp = info.get("enemy_hp", 176)
        return obs, info
    
    def render(self, mode="human"):
        return self.game.render()
    
    def close(self):
        self.game.close()


def play_street_fighter(
    model_path: str = "/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-7B-Instruct-AWQ",
    episodes: int = 3,
    render: bool = True,
    verbose: bool = True,
):
    """
    Play Street Fighter 2 with Qwen agent

    Args:
        model_path: Path to Qwen model
        episodes: Number of episodes to play
        render: Whether to render the game
        verbose: Whether to print detailed reasoning
    """
    print("🥊 Starting Street Fighter 2 Gameplay")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print("-" * 50)

    # Create environment using the existing wrapper
    env = StreetFighter()
    
    # Create agent
    agent = QwenStreetFighterAgent(model_path)

    total_rewards = []
    wins = 0

    for episode in range(episodes):
        print(f"\n🏁 Episode {episode + 1}/{episodes}")
        obs = env.reset()
        agent.reset()
        
        total_reward = 0
        step = 0
        max_steps = 2000
        agent_won = False

        while step < max_steps:
            if render:
                env.render()

            # Mock info (replace with actual game state extraction if available)
            info = {
                'agent_hp': 150,
                'enemy_hp': 120,
                'agent_x': 100 + (step % 50),
                'agent_y': 200,
                'enemy_x': 200 - (step % 30),
                'enemy_y': 200,
                'enemy_status': 0,
            }

            # Get action from agent
            action, reasoning = agent.get_action(obs, info, verbose=verbose)

            # Take step in environment (wrapper returns 5 values)
            obs, reward, done, truncated, info_dict = env.step(action)
            total_reward += reward

            # Check for episode end
            if done or step >= max_steps:
                # Simple win detection (you may need to improve this)
                if reward > 0:
                    agent_won = True
                    wins += 1
                break

            step += 1

        total_rewards.append(total_reward)
        
        status = "🏆 WON" if agent_won else "💀 LOST"
        print(f"Episode {episode + 1} completed: {status} | Reward: {total_reward:.2f} | Steps: {step}")

    env.close()

    # Print summary
    print("\n" + "=" * 50)
    print("🏁 GAMEPLAY SUMMARY")
    print("=" * 50)
    print(f"Episodes Played: {len(total_rewards)}")
    print(f"Wins: {wins}")
    print(f"Win Rate: {wins/len(total_rewards)*100:.1f}%" if total_rewards else "N/A")
    print(f"Average Reward: {sum(total_rewards)/len(total_rewards):.2f}" if total_rewards else "N/A")


def main():
    parser = argparse.ArgumentParser(
        description="Street Fighter 2 Gameplay with Qwen Agent"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-7B-Instruct-AWQ",
        help="Qwen2.5-VL model path",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to play",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering for faster execution",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose reasoning output",
    )
    
    args = parser.parse_args()

    play_street_fighter(
        model_path=args.model,
        episodes=args.episodes,
        render=not args.no_render,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()