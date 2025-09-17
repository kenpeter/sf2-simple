#!/usr/bin/env python3
"""
Online LoRA Training Script for Qwen2.5-VL Street Fighter Agent
Real-time learning during gameplay with experience replay
"""

import torch
import numpy as np
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse
from typing import List, Dict, Tuple, Deque
import json
import os
from PIL import Image
import random
from collections import deque
import threading
import time
import glob

# Import game environment
import retro
import gymnasium as gym
from gymnasium import spaces
import cv2
from discretizer import StreetFighter2Discretizer
from qwen_agent import QwenStreetFighterAgent


def find_latest_checkpoint(output_dir: str = "./sf2_online_lora") -> str:
    """Find the latest checkpoint in the output directory"""
    if not os.path.exists(output_dir):
        return None

    # Find all checkpoint directories (both episode_* and checkpoint_*)
    episode_pattern = os.path.join(output_dir, "episode_*")
    checkpoint_pattern = os.path.join(output_dir, "checkpoint_*")
    episodes = glob.glob(episode_pattern)
    checkpoints = glob.glob(checkpoint_pattern)

    # Prefer episode checkpoints over update checkpoints
    if episodes:
        episode_nums = []
        for ep in episodes:
            try:
                num = int(os.path.basename(ep).split("_")[1])
                episode_nums.append((num, ep))
            except (IndexError, ValueError):
                continue

        if episode_nums:
            # Return the episode with highest number
            latest_episode = max(episode_nums, key=lambda x: x[0])[1]
            return latest_episode

    # Fallback to old checkpoint naming
    if checkpoints:
        checkpoint_nums = []
        for cp in checkpoints:
            try:
                num = int(os.path.basename(cp).split("_")[1])
                checkpoint_nums.append((num, cp))
            except (IndexError, ValueError):
                continue

        if checkpoint_nums:
            latest_checkpoint = max(checkpoint_nums, key=lambda x: x[0])[1]
            return latest_checkpoint

    return None


def list_checkpoints(output_dir: str = "./sf2_online_lora"):
    """List all available checkpoints"""
    if not os.path.exists(output_dir):
        print(f"❌ Directory {output_dir} does not exist")
        return

    # Find both episode and checkpoint patterns
    episode_pattern = os.path.join(output_dir, "episode_*")
    checkpoint_pattern = os.path.join(output_dir, "checkpoint_*")
    episodes = glob.glob(episode_pattern)
    checkpoints = glob.glob(checkpoint_pattern)

    all_checkpoints = episodes + checkpoints

    if not all_checkpoints:
        print(f"📁 No checkpoints found in {output_dir}")
        return

    print(f"📁 Available checkpoints in {output_dir}:")

    checkpoint_info = []

    # Process episode checkpoints
    for ep in episodes:
        try:
            num = int(os.path.basename(ep).split("_")[1])

            # Try to load training state
            state_file = os.path.join(ep, "training_state.json")
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state = json.load(f)
                loss = state.get("avg_recent_loss", "N/A")
                updates = state.get("total_updates", "N/A")
                info = (
                    f"(Loss: {loss:.4f}, Updates: {updates})" if loss != "N/A" else ""
                )
            else:
                info = "(No training state)"

            checkpoint_info.append((num, f"episode_{num}", ep, info, "episode"))
        except (IndexError, ValueError):
            continue

    # Process old checkpoint format
    for cp in checkpoints:
        try:
            num = int(os.path.basename(cp).split("_")[1])

            # Try to load training state
            state_file = os.path.join(cp, "training_state.json")
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state = json.load(f)
                loss = state.get("avg_recent_loss", "N/A")
                lr = state.get("learning_rate", "N/A")
                info = f"(Loss: {loss:.4f}, LR: {lr})" if loss != "N/A" else ""
            else:
                info = "(No training state)"

            checkpoint_info.append((num, f"checkpoint_{num}", cp, info, "checkpoint"))
        except (IndexError, ValueError):
            continue

    # Sort by number, episodes first
    checkpoint_info.sort(key=lambda x: (x[4] == "checkpoint", x[0]))

    for num, name, path, info, type_name in checkpoint_info:
        print(f"  • {name}: {path} {info}")

    if checkpoint_info:
        # Find latest episode if available, otherwise latest checkpoint
        episodes_only = [x for x in checkpoint_info if x[4] == "episode"]
        if episodes_only:
            latest = max(episodes_only, key=lambda x: x[0])
            print(f"\n💡 Latest episode: {latest[1]}")
            print(f"💡 To resume: --resume-from {latest[2]}")
        else:
            latest = max(checkpoint_info, key=lambda x: x[0])
            print(f"\n💡 Latest checkpoint: {latest[1]}")
            print(f"💡 To resume: --resume-from {latest[2]}")


class OnlineExperienceBuffer:
    """Experience replay buffer for online learning"""

    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add_experience(
        self,
        image: Image.Image,
        game_state: Dict,
        action: int,
        reward: float,
        next_image: Image.Image,
        next_state: Dict,
        done: bool,
    ):
        """Add a new experience to the buffer"""
        experience = {
            "image": image,
            "game_state": game_state,
            "action": action,
            "reward": reward,
            "next_image": next_image,
            "next_state": next_state,
            "done": done,
            "timestamp": time.time(),
        }
        self.buffer.append(experience)

    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample a random batch of experiences"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)

    def get_recent_experiences(self, num_recent: int) -> List[Dict]:
        """Get the most recent experiences"""
        return list(self.buffer)[-num_recent:]

    def __len__(self):
        return len(self.buffer)


class OnlineLoRATrainer:
    """Online LoRA trainer for real-time learning during gameplay"""

    def __init__(
        self,
        model_path: str = "/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-3B-Instruct",
        output_dir: str = "./sf2_online_lora",
        lora_rank: int = 16,  # Higher rank for better learning capacity
        lora_alpha: int = 32,  # Higher alpha for stronger adaptation
        lora_dropout: float = 0.1,
        experience_buffer_size: int = 100,  # Reduced buffer size
        learning_rate: float = 5e-5,
        resume_from: str = None,  # Path to checkpoint to resume from
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.learning_rate = learning_rate

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize experience buffer
        self.experience_buffer = OnlineExperienceBuffer(experience_buffer_size)

        # Initialize model and processor
        print(f"🤖 Loading base model for online training: {model_path}")
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )

        # Fix tokenizer if needed
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # Load model
        print("📍 Loading model for online training...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=True,
        )

        # Configure LoRA for online learning
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        # Apply LoRA or load from checkpoint
        if resume_from and os.path.exists(resume_from):
            print(f"🔄 Resuming from checkpoint: {resume_from}")
            # Load the LoRA adapter from checkpoint
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, resume_from)
            print("✅ LoRA checkpoint loaded successfully")

            # Enable training mode and ensure gradients are enabled
            self.model.train()

            # Re-enable LoRA adapters specifically
            if hasattr(self.model, "enable_adapters"):
                try:
                    self.model.enable_adapters()
                except ValueError:
                    # Adapter already enabled, continue
                    pass

            # Ensure gradients are enabled for LoRA parameters
            for name, param in self.model.named_parameters():
                if "lora_" in name.lower() or "adapter" in name.lower():
                    param.requires_grad = True

            # Try to load training state
            state_file = os.path.join(resume_from, "training_state.json")
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    training_state = json.load(f)
                self.total_updates = training_state.get("total_updates", 0)
                print(f"📊 Resuming from update #{self.total_updates}")
            else:
                self.total_updates = 0
                print("⚠️ No training state found, starting update count from 0")
        else:
            # Apply fresh LoRA
            self.model = get_peft_model(self.model, lora_config)
            print(f"✅ Online LoRA applied with rank {lora_rank}, alpha {lora_alpha}")
            self.total_updates = 0

        self.model.print_trainable_parameters()

        # Training state
        self.training_active = False
        self.training_thread = None

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Performance tracking
        self.episode_rewards = []
        self.update_losses = []

        # Action tracking for diversity
        self.recent_actions = []
        self.action_counts = {}  # Track action frequency

    def create_training_prompt(self, game_state: Dict) -> str:
        """Create training prompt from game state with action diversity tracking"""

        # Get recent actions for diversity tracking only
        recent_actions_str = ""
        if len(self.recent_actions) > 0:
            recent_actions_str = f"\nRecent actions used: {self.recent_actions[-5:]}"

        prompt = f"""Street Fighter 2 Game State:

Current Situation:
- Agent HP: {game_state.get('agent_hp', 176)}
- Enemy HP: {game_state.get('enemy_hp', 176)}
- Distance: {game_state.get('distance', 100)}px
- Agent Position: ({game_state.get('agent_x', 0)}, {game_state.get('agent_y', 0)})
- Enemy Position: ({game_state.get('enemy_x', 0)}, {game_state.get('enemy_y', 0)}){recent_actions_str}

Available actions: 0=NO_ACTION, 1=UP, 2=DOWN, 3=LEFT, 6=RIGHT, 9=LIGHT_PUNCH, 13=MEDIUM_PUNCH, 17=HEAVY_PUNCH, 21=LIGHT_KICK, 26=MEDIUM_KICK, 32=HEAVY_KICK, 38=HADOKEN_RIGHT, 39=DRAGON_PUNCH_RIGHT, 40=HURRICANE_KICK_RIGHT, 41=HADOKEN_LEFT, 42=DRAGON_PUNCH_LEFT, 43=HURRICANE_KICK_LEFT

Choose optimal action (0-43):"""
        return prompt

    def process_experience_for_training(self, experience: Dict) -> Dict:
        """Convert experience to training format"""

        # Track recent actions for diversity
        action = experience["action"]
        self.recent_actions.append(action)
        if len(self.recent_actions) > 10:  # Keep last 10 actions
            self.recent_actions = self.recent_actions[-10:]

        # Create prompt
        prompt = self.create_training_prompt(experience["game_state"])

        # Create conversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": experience["image"]},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": str(experience["action"])},
        ]

        # TEXT-ONLY training to avoid image token issues
        response_text = str(experience["action"])

        # Create detailed text description instead of using image
        game_state = experience["game_state"]
        text_description = f"""Street Fighter 2 Game State:
- Agent HP: {game_state.get('agent_hp', 176)}
- Enemy HP: {game_state.get('enemy_hp', 176)}
- Distance: {abs(game_state.get('agent_x', 0) - game_state.get('enemy_x', 0))}px
- Agent Position: ({game_state.get('agent_x', 0)}, {game_state.get('agent_y', 0)})
- Enemy Position: ({game_state.get('enemy_x', 0)}, {game_state.get('enemy_y', 0)})

Optimal action: {response_text}"""

        inputs = self.processor(
            text=text_description,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": inputs["input_ids"].squeeze(),
            "pixel_values": (
                inputs.get("pixel_values", torch.empty(0)).squeeze()
                if inputs.get("pixel_values") is not None
                else None
            ),
            "image_grid_thw": (
                inputs.get("image_grid_thw", torch.empty(0)).squeeze()
                if inputs.get("image_grid_thw") is not None
                else None
            ),
        }

    def perform_online_update(
        self, batch_size: int = 1
    ):  # Reduced batch size for memory
        """Perform a single online learning update"""
        if len(self.experience_buffer) < batch_size:
            return None

        # Sample experiences
        experiences = self.experience_buffer.sample_batch(batch_size)

        # Process experiences for training
        processed_data = []
        for exp in experiences:
            try:
                processed = self.process_experience_for_training(exp)
                processed_data.append(processed)
            except Exception as e:
                print(f"⚠️ Error processing experience: {e}")
                continue

        if not processed_data:
            return None

        # Custom data collator
        def collate_batch(batch):
            # Handle tensor stacking with proper error checking
            try:
                input_ids = torch.stack(
                    [
                        (
                            torch.tensor(item["input_ids"], dtype=torch.long)
                            if not isinstance(item["input_ids"], torch.Tensor)
                            else item["input_ids"].long()
                        )
                        for item in batch
                    ]
                )

                attention_mask = torch.stack(
                    [
                        (
                            torch.tensor(item["attention_mask"], dtype=torch.long)
                            if not isinstance(item["attention_mask"], torch.Tensor)
                            else item["attention_mask"].long()
                        )
                        for item in batch
                    ]
                )

                labels = torch.stack(
                    [
                        (
                            torch.tensor(item["labels"], dtype=torch.long)
                            if not isinstance(item["labels"], torch.Tensor)
                            else item["labels"].long()
                        )
                        for item in batch
                    ]
                )

                result = {
                    "input_ids": input_ids.cuda(),
                    "attention_mask": attention_mask.cuda(),
                    "labels": labels.cuda(),
                }

                # Add pixel values if present
                if batch[0]["pixel_values"] is not None:
                    pixel_values = torch.stack(
                        [
                            (
                                torch.tensor(item["pixel_values"], dtype=torch.float16)
                                if not isinstance(item["pixel_values"], torch.Tensor)
                                else item["pixel_values"].half()
                            )
                            for item in batch
                            if item["pixel_values"] is not None
                        ]
                    )
                    result["pixel_values"] = pixel_values.cuda()

                # Add image grid thw if present
                if batch[0]["image_grid_thw"] is not None:
                    image_grid_thw = torch.stack(
                        [
                            (
                                torch.tensor(item["image_grid_thw"], dtype=torch.long)
                                if not isinstance(item["image_grid_thw"], torch.Tensor)
                                else item["image_grid_thw"].long()
                            )
                            for item in batch
                            if item["image_grid_thw"] is not None
                        ]
                    )
                    result["image_grid_thw"] = image_grid_thw.cuda()

                return result
            except Exception as e:
                print(f"❌ Error in batch collation: {e}")
                return None

        # Perform forward pass and compute loss
        try:
            batch = collate_batch(processed_data)
            if batch is None:
                return None

            self.model.train()

            # Forward pass with memory optimization
            with torch.amp.autocast("cuda"):
                outputs = self.model(**batch)
                loss = outputs.loss

            # Backward pass
            loss.backward()

            # Update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.total_updates += 1
            loss_value = loss.item()
            self.update_losses.append(loss_value)

            # Keep only recent losses
            if len(self.update_losses) > 100:
                self.update_losses = self.update_losses[-100:]

            print(f"🔄 Online update #{self.total_updates}: Loss = {loss_value:.4f}")

            return loss_value

        except Exception as e:
            print(f"❌ Error in online update: {e}")
            return None

    def save_model(self, episode_num=None):
        """Save the current model state with training information"""
        if episode_num is not None:
            save_path = os.path.join(self.output_dir, f"episode_{episode_num}")
        else:
            save_path = os.path.join(
                self.output_dir, f"checkpoint_{self.total_updates}"
            )

        os.makedirs(save_path, exist_ok=True)

        # Save the LoRA adapter
        self.model.save_pretrained(save_path)

        # Save training state
        training_state = {
            "total_updates": self.total_updates,
            "episode_num": episode_num,
            "avg_recent_loss": (
                np.mean(self.update_losses[-10:]) if self.update_losses else 0.0
            ),
            "buffer_size": len(self.experience_buffer),
            "learning_rate": self.learning_rate,
        }

        state_file = os.path.join(save_path, "training_state.json")
        with open(state_file, "w") as f:
            json.dump(training_state, f, indent=2)

        print(f"💾 Model and training state saved to {save_path}")
        return save_path

    def get_training_stats(self) -> Dict:
        """Get current training statistics"""
        return {
            "total_updates": self.total_updates,
            "buffer_size": len(self.experience_buffer),
            "avg_recent_loss": (
                np.mean(self.update_losses[-10:]) if self.update_losses else 0.0
            ),
            "avg_episode_reward": (
                np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
            ),
        }


# the game env
class OnlineStreetFighterEnv(gym.Env):
    """Street Fighter environment for online learning"""

    # init
    def __init__(self):
        # super init
        super().__init__()
        # retro make game
        # retro load state
        # retro strict action
        game = retro.make(
            "StreetFighterIISpecialChampionEdition-Genesis",
            state="ken_bison_12.state",
            use_restricted_actions=retro.Actions.FILTERED,
        )

        # combo
        self.game = StreetFighter2Discretizer(game)

        # obs space box
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(224, 320, 3), dtype=np.uint8
        )

        # action space
        self.action_space = self.game.action_space

        # hp
        self.agent_hp = 176
        self.enemy_hp = 176

    # raw game frame
    def get_raw_frame(self):
        """Get raw RGB frame for vision model"""
        try:
            raw_obs = self.game.env.unwrapped.render()
            if raw_obs is not None and isinstance(raw_obs, np.ndarray):
                if len(raw_obs.shape) == 3 and raw_obs.shape[2] == 3:
                    resized = cv2.resize(
                        raw_obs, (320, 224), interpolation=cv2.INTER_CUBIC
                    )
                    return Image.fromarray(resized)
        except:
            pass
        # Fallback
        return Image.fromarray(np.zeros((224, 320, 3), dtype=np.uint8))

    def step(self, action):
        obs, _, done, truncated, info = self.game.step(action)

        current_agent_hp = info.get("agent_hp", self.agent_hp)
        current_enemy_hp = info.get("enemy_hp", self.enemy_hp)

        # HP-based reward with special move encouragement
        delta_hp_diff = (current_agent_hp - self.agent_hp) - (
            current_enemy_hp - self.enemy_hp
        )
        reward = delta_hp_diff / 176.0 - 0.0001

        # Bonus for special moves (non-rule based, just incentive)
        special_moves = [
            38,
            39,
            40,
            41,
            42,
            43,
        ]  # Hadoken, Dragon Punch, Hurricane Kick
        if action in special_moves:
            reward += 0.01  # Significant bonus for trying special moves

        self.agent_hp = current_agent_hp
        self.enemy_hp = current_enemy_hp

        # Check for round end
        if self.agent_hp <= 0 or self.enemy_hp <= 0:
            done = True
            if self.enemy_hp <= 0 and self.agent_hp > 0:
                reward += 1.0
                info["agent_won"] = True
            else:
                reward -= 1.0
                info["agent_won"] = False

        # Add distance calculation
        info["distance"] = abs(info.get("agent_x", 100) - info.get("enemy_x", 200))

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        result = self.game.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        self.agent_hp = info.get("agent_hp", 176)
        self.enemy_hp = info.get("enemy_hp", 176)
        return obs, info

    def render(self, mode="human"):
        return self.game.render()

    def close(self):
        self.game.close()


def run_online_training(
    model_path: str,
    episodes: int = 10,
    learning_rate: float = 5e-5,
    update_frequency: int = 30,  # Less frequent updates
    save_frequency: int = 1,  # Save every N episodes
    render: bool = False,
    resume_from: str = None,  # Resume from checkpoint
):
    """Run online training with real-time learning"""

    print("🎮 Starting Online LoRA Training")
    print(f"Model: {model_path}")
    if resume_from:
        print(f"🔄 Resuming from: {resume_from}")
    print(f"Episodes: {episodes}")
    print(f"Learning Rate: {learning_rate}")
    print("-" * 50)

    # Initialize trainer first
    trainer = OnlineLoRATrainer(
        model_path=model_path,
        learning_rate=learning_rate,
        resume_from=resume_from,
    )

    # Initialize environment
    env = OnlineStreetFighterEnv()

    # Create a lightweight agent that uses the trainer's model
    class LightweightAgent:
        def __init__(self, trainer):
            self.trainer = trainer
            self.frame_counter = 0
            self.last_action = 0
            self.action_cooldown = 0

        def get_action(self, obs, info, verbose=False):
            # frame counter
            self.frame_counter += 1

            # Use the actual trained LoRA model for inference
            game_state = info.copy()

            # Create text description for the model with exploration encouragement
            text_description = f"""Fighting Game State:
- Agent HP: {game_state.get('agent_hp', 176)}
- Enemy HP: {game_state.get('enemy_hp', 176)}
- Distance: {abs(game_state.get('agent_x', 0) - game_state.get('enemy_x', 0))}px
- Agent Position: ({game_state.get('agent_x', 0)}, {game_state.get('agent_y', 0)})
- Enemy Position: ({game_state.get('enemy_x', 0)}, {game_state.get('enemy_y', 0)})

Explore different fighting strategies! Choose optimal action (0-43):"""

            try:
                # Use the trained model for inference
                inputs = self.trainer.processor(
                    text=text_description,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )

                # Move to device
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

                # Generate action with natural exploration via sampling
                with torch.no_grad():
                    outputs = self.trainer.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=True,          # Enable sampling for exploration
                        temperature=0.8,         # Add controlled randomness
                        top_p=0.9,              # Nucleus sampling
                        top_k=20,               # Consider top 20 tokens
                        pad_token_id=self.trainer.processor.tokenizer.pad_token_id,
                    )

                # Decode response
                response = self.trainer.processor.decode(
                    outputs[0], skip_special_tokens=True
                )

                # Extract action number from response using sophisticated parsing
                import re

                # Look for standalone numbers first (most direct answer)
                standalone_number = re.search(r"^\s*(\d+)\s*$", response, re.MULTILINE)
                if standalone_number:
                    action = int(standalone_number.group(1))
                    if 0 <= action <= 43:
                        return action, f"LoRA standalone: {action}"

                # Look for "Action: X" or "Action X" pattern
                action_match = re.search(
                    r"(?:Action:?\s*|^)(\d+)", response, re.IGNORECASE | re.MULTILINE
                )
                if action_match:
                    action = int(action_match.group(1))
                    if 0 <= action <= 43:
                        return action, f"LoRA action pattern: {action}"

                # Look for numbers but filter out context-specific ones
                numbers = re.findall(r"\b(\d+)\b", response)
                for num_str in numbers:
                    num = int(num_str)
                    # Skip numbers likely from "Street Fighter 2" or other context
                    if num == 2 and "Fighter 2" in response:
                        continue
                    if 0 <= num <= 43:
                        return num, f"LoRA filtered: {num}"

                # Enhanced fallback with random exploration
                fallback_action = random.randint(0, 43)
                return fallback_action, f"LoRA random fallback: {fallback_action}"

            except Exception as e:
                # Simple fallback for errors
                return 0, f"Error: {e}"

        def reset(self):
            self.frame_counter = 0
            self.last_action = 0
            self.action_cooldown = 0

    agent = LightweightAgent(trainer)

    total_steps = 0

    for episode in range(episodes):
        print(f"\n🏁 Episode {episode + 1}/{episodes}")

        obs, info = env.reset()
        agent.reset()

        episode_reward = 0
        step = 0
        max_steps = 5000
        prev_image = env.get_raw_frame()
        prev_info = info.copy()

        while step < max_steps:
            if render:
                env.render()

            # Get current game state
            current_image = env.get_raw_frame()

            # Get action from agent
            action, reasoning = agent.get_action(obs, info, verbose=False)
            
            # Log reasoning for debugging (only occasionally to avoid spam)
            if total_steps % 100 == 0:
                print(f"    Reasoning: {reasoning}")

            # Take step in environment
            obs, reward, done, truncated, next_info = env.step(action)
            next_image = env.get_raw_frame()

            # Add experience to buffer
            trainer.experience_buffer.add_experience(
                image=prev_image,
                game_state=prev_info,
                action=action,
                reward=reward,
                next_image=next_image,
                next_state=next_info,
                done=done,
            )

            # Perform online learning update (less frequent, smaller batch)
            if (
                total_steps % update_frequency == 0
                and len(trainer.experience_buffer) >= 1
            ):
                # Clear GPU cache before training
                torch.cuda.empty_cache()
                loss = trainer.perform_online_update(batch_size=1)
                if loss is not None:
                    print(
                        f"  Step {step}: Action {action}, Reward {reward:.3f}, Loss {loss:.4f}"
                    )
                # Clear cache again after training
                torch.cuda.empty_cache()

            # Save model will be handled after each episode instead

            episode_reward += reward
            step += 1
            total_steps += 1

            # Update for next iteration
            prev_image = current_image
            prev_info = next_info.copy()

            if done or truncated:
                break

        # Record episode performance
        trainer.episode_rewards.append(episode_reward)

        # Save model after each episode (if save_frequency episodes have passed)
        if (episode + 1) % save_frequency == 0:
            trainer.save_model(episode_num=episode + 1)
            print(f"💾 Episode checkpoint saved: episode_{episode + 1}")

        # Print episode summary
        agent_won = next_info.get("agent_won", False)
        stats = trainer.get_training_stats()
        print(f"Episode {episode + 1} completed:")
        print(f"  Result: {'🏆 WON' if agent_won else '💀 LOST'}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {step}")
        print(f"  Buffer: {stats['buffer_size']} experiences")
        print(f"  Updates: {stats['total_updates']}")
        print(f"  Avg Loss: {stats['avg_recent_loss']:.4f}")

    env.close()

    # Final save
    trainer.save_model(episode_num=episodes)

    # Print final statistics
    print("\n" + "=" * 50)
    print("🏁 ONLINE TRAINING COMPLETED")
    print("=" * 50)
    final_stats = trainer.get_training_stats()
    print(f"Total Episodes: {episodes}")
    print(f"Total Updates: {final_stats['total_updates']}")
    print(f"Final Buffer Size: {final_stats['buffer_size']}")
    print(f"Average Recent Loss: {final_stats['avg_recent_loss']:.4f}")
    print(f"Average Episode Reward: {final_stats['avg_episode_reward']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Online LoRA Training for Street Fighter Qwen Agent"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-3B-Instruct",
        help="Base model path",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to train"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--update-frequency",
        type=int,
        default=50,
        help="Perform learning update every N steps",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="Save model every N episodes"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the game during training"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint directory (e.g., ./sf2_online_lora/checkpoint_40)",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume from the latest checkpoint",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List all available checkpoints and exit",
    )

    args = parser.parse_args()

    # Handle special commands
    if args.list_checkpoints:
        list_checkpoints()
        return

    # Handle auto-resume
    resume_from = args.resume_from
    if args.auto_resume and not resume_from:
        resume_from = find_latest_checkpoint()
        if resume_from:
            print(f"🔄 Auto-resuming from latest checkpoint: {resume_from}")
        else:
            print("⚠️ No checkpoints found for auto-resume, starting fresh training")

    run_online_training(
        model_path=args.model,
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        update_frequency=args.update_frequency,
        save_frequency=args.save_frequency,
        render=args.render,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    main()
