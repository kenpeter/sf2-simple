#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Qwen2.5-VL Street Fighter Agent
Fine-tune the vision-language model for better Street Fighter gameplay
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
from datasets import Dataset
import argparse
from typing import List, Dict, Tuple
import json
import os
from PIL import Image
import random
import base64
from io import BytesIO


class StreetFighterDataset:
    """Dataset for Street Fighter gameplay data"""

    def __init__(self, data_path: str = None, shuffle=True, sample_size=None):
        self.data = []
        if data_path is not None and os.path.exists(data_path):
            self.load_data(data_path, shuffle=shuffle, sample_size=sample_size)

    def load_data(self, data_path: str, shuffle=True, sample_size=None):
        """Load training data from JSON file with base64 image decoding"""
        with open(data_path, "r") as f:
            data = json.load(f)

        # Random sampling options for better training diversity
        if sample_size and len(data) > sample_size:
            print(f"🎲 Randomly sampling {sample_size} from {len(data)} examples")
            data = random.sample(data, sample_size)
        
        if shuffle:
            print(f"🔀 Shuffling {len(data)} training examples")
            random.shuffle(data)

        # Convert base64 images back to PIL Images
        self.data = []
        for item in data:
            if isinstance(item["image"], str):  # base64 encoded
                # Decode base64 image
                image_data = base64.b64decode(item["image"])
                image = Image.open(BytesIO(image_data))
                item["image"] = image
            self.data.append(item)

        print(f"📁 Loaded {len(self.data)} training examples (randomized: {shuffle})")

    def add_example(
        self, image: Image.Image, game_state: Dict, action: int, reasoning: str
    ):
        """Add a training example"""
        example = {
            "image": image,  # Keep as PIL Image for processing
            "game_state": game_state,
            "action": action,
            "reasoning": reasoning,
        }
        self.data.append(example)

    def save_data(self, data_path: str, append: bool = False):
        """Save training data to JSON file with base64 image encoding"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        # Load existing data if appending
        existing_data = []
        if append and os.path.exists(data_path):
            try:
                with open(data_path, "r") as f:
                    existing_data = json.load(f)
                print(f"📁 Found {len(existing_data)} existing training examples")
            except (json.JSONDecodeError, FileNotFoundError):
                print("⚠️ Could not load existing data, creating new file")
                existing_data = []

        # Convert current data to JSON-serializable format
        serializable_data = []
        for item in self.data:
            # Convert PIL Image to base64
            if isinstance(item["image"], Image.Image):
                buffered = BytesIO()
                item["image"].save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_data = img_base64
            else:
                image_data = item["image"]  # Already encoded

            serializable_item = {
                "image": image_data,
                "game_state": item["game_state"],
                "action": item["action"],
                "reasoning": item["reasoning"],
            }
            serializable_data.append(serializable_item)

        # Combine with existing data if appending
        if append:
            all_data = existing_data + serializable_data
            print(
                f"📊 Appending {len(serializable_data)} new examples to {len(existing_data)} existing ones"
            )
        else:
            all_data = serializable_data

        # Save to JSON file
        with open(data_path, "w") as f:
            json.dump(all_data, f, indent=2)

        print(f"💾 Saved {len(all_data)} total training examples to {data_path}")

    def create_prompt(self, game_state: Dict) -> str:
        """Create training prompt from game state"""
        prompt = f"""STREET FIGHTER 2 ANALYSIS

GAME STATE:
- My HP: {game_state.get('agent_hp', 176)} | Enemy HP: {game_state.get('enemy_hp', 176)}
- Distance: {game_state.get('distance', 100)}px
- My Position: ({game_state.get('agent_x', 0)}, {game_state.get('agent_y', 0)})
- Enemy Position: ({game_state.get('enemy_x', 0)}, {game_state.get('enemy_y', 0)})
- Enemy Status: {game_state.get('enemy_status', 0)}

Available actions: 0=NO_ACTION, 1=UP, 2=DOWN, 3=LEFT, 6=RIGHT, 9=LIGHT_PUNCH, 13=MEDIUM_PUNCH, 17=HEAVY_PUNCH, 21=LIGHT_KICK, 26=MEDIUM_KICK, 32=HEAVY_KICK, 38=HADOKEN_RIGHT, 39=DRAGON_PUNCH_RIGHT, 40=HURRICANE_KICK_RIGHT, 41=HADOKEN_LEFT, 42=DRAGON_PUNCH_LEFT, 43=HURRICANE_KICK_LEFT

Return only the action number (0-43):"""
        return prompt


class StreetFighterLoRATrainer:
    """LoRA trainer for Street Fighter Qwen agent"""

    def __init__(
        self,
        model_path: str = "/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-3B-Instruct-AWQ",
        output_dir: str = "./sf2_lora_model",
        # lora rank?
        lora_rank: int = 16,
        # lora alpha
        lora_alpha: int = 32,
        # lora dropout: dropout
        lora_dropout: float = 0.1,
    ):
        self.model_path = model_path
        self.output_dir = output_dir

        # Initialize model and processor
        print(f"🤖 Loading base model: {model_path}")
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )

        # Fix tokenizer if needed
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # Load model on GPU (AWQ models don't support CPU offloading)
        print("📍 Loading model on GPU...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=True,
        )

        """
        
            Detailed flow:

            Input (hidden states)
                ↓
            ┌─ q_proj → Query vectors ─┐
            ├─ k_proj → Key vectors   ─┤
            └─ v_proj → Value vectors ─┘
                ↓
            Attention computation (Q·K·V)
                ↓
            o_proj → transform attention output
                ↓
            Add & Norm (residual connection)
                ↓
            ┌─ gate_proj → SiLU activation ─┐
            └─ up_proj → higher dimension  ─┘
                ↓
            Element-wise multiply (gate * up)
                ↓
            down_proj → back to original dimension
                ↓
            Add & Norm (residual connection)
                ↓
            Output (to next transformer block)

        """

        # Configure LoRA
        lora_config = LoraConfig(
            # model tpye
            task_type=TaskType.CAUSAL_LM,
            # how much we get from original matrix
            r=lora_rank,
            # lora_alpha / lora_rank, how much inference to the raw lora correctness
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

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        print(f"✅ LoRA applied with rank {lora_rank}, alpha {lora_alpha}")

        # Print trainable parameters
        self.model.print_trainable_parameters()

    def prepare_dataset(self, dataset: StreetFighterDataset) -> Dataset:
        """Prepare dataset for training"""

        def process_example(example):
            # Create prompt
            prompt = dataset.create_prompt(example["game_state"])

            # Create full conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": prompt},
                    ],
                },
                {"role": "assistant", "content": str(example["action"])},
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Process inputs using the same method as qwen_agent.py
            # Convert PIL image to format expected by processor
            image = example["image"]
            if hasattr(image, "convert"):
                image = image.convert("RGB")

            # Use the processor with proper image handling
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            result = {
                "input_ids": inputs["input_ids"].squeeze(),
                "attention_mask": inputs["attention_mask"].squeeze(),
                "labels": inputs[
                    "input_ids"
                ].squeeze(),  # For causal LM, labels = input_ids
            }

            # Add all necessary image-related tensors
            if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                result["pixel_values"] = inputs["pixel_values"].squeeze()

            if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
                result["image_grid_thw"] = inputs["image_grid_thw"].squeeze()

            return result

        # Process all examples
        processed_data = [process_example(example) for example in dataset.data]

        return Dataset.from_list(processed_data)

    def train(
        self,
        dataset: StreetFighterDataset,
        num_epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        save_steps: int = 100,
        resume_from_checkpoint: str = None,
    ):
        """Train the LoRA model"""
        # Prepare dataset
        train_dataset = self.prepare_dataset(dataset)

        # Training arguments - optimized for GPU memory with better progress tracking
        training_args = TrainingArguments(
            # output model dir
            output_dir=self.output_dir,
            # how many iteration
            num_train_epochs=num_epochs,
            # batch size can be 1 not too large
            per_device_train_batch_size=batch_size,
            # load 1 batch to GPU, then process, 16x1
            gradient_accumulation_steps=16,
            # learning rate
            learning_rate=learning_rate,
            # wegith decay
            weight_decay=0.01,
            # Progress tracking - log more frequently for better feedback
            logging_steps=5,  # Log every 5 steps instead of 10
            logging_first_step=True,
            logging_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,  # Reduce saved checkpoints
            warmup_steps=20,  # Reduce warmup steps
            fp16=True,
            gradient_checkpointing=False,  # Disable - causes issues with LoRA
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Reduce memory overhead
            optim="adamw_torch",  # Use more memory efficient optimizer
            # Better progress reporting
            report_to=None,  # Disable wandb/tensorboard for cleaner output
            disable_tqdm=False,  # Keep progress bars
            log_level="info",
            deepspeed="ds_config.json" if os.path.exists("ds_config.json") else None,
        )

        # Custom data collator with proper tensor handling
        def data_collator(batch):
            # Convert all tensors and ensure proper dtypes
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
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            # Add pixel values if present
            if "pixel_values" in batch[0] and batch[0]["pixel_values"] is not None:
                pixel_values = torch.stack(
                    [
                        (
                            torch.tensor(item["pixel_values"], dtype=torch.float16)
                            if not isinstance(item["pixel_values"], torch.Tensor)
                            else item["pixel_values"].half()
                        )
                        for item in batch
                    ]
                )
                result["pixel_values"] = pixel_values

            # Add image grid thw if present
            if "image_grid_thw" in batch[0] and batch[0]["image_grid_thw"] is not None:
                image_grid_thw = torch.stack(
                    [
                        (
                            torch.tensor(item["image_grid_thw"], dtype=torch.long)
                            if not isinstance(item["image_grid_thw"], torch.Tensor)
                            else item["image_grid_thw"].long()
                        )
                        for item in batch
                    ]
                )
                result["image_grid_thw"] = image_grid_thw

            return result

        # Custom callback for better progress tracking
        from transformers import TrainerCallback
        
        class ProgressCallback(TrainerCallback):
            def __init__(self, total_steps):
                self.total_steps = total_steps
                self.start_time = None
                
            def on_train_begin(self, args, state, control, **kwargs):
                import time
                self.start_time = time.time()
                print(f"🎯 Training {len(train_dataset)} examples for {num_epochs} epochs")
                print(f"📊 Total steps: {self.total_steps} (batch_size={batch_size}, grad_accum={training_args.gradient_accumulation_steps})")
                
            def on_log(self, args, state, control, logs=None, **kwargs):
                import time
                if logs:
                    current_step = state.global_step
                    progress_pct = (current_step / self.total_steps) * 100
                    
                    # Format loss
                    loss = logs.get('train_loss', 0)
                    lr = logs.get('learning_rate', learning_rate)
                    
                    # Calculate ETA - handle division by zero
                    if self.start_time and current_step > 0 and self.total_steps > 0:
                        elapsed = time.time() - self.start_time
                        steps_per_sec = current_step / elapsed
                        remaining_steps = self.total_steps - current_step
                        eta_sec = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                        eta_min = eta_sec / 60
                        eta_str = f"{eta_min:.1f}min" if eta_min > 1 else f"{eta_sec:.0f}s"
                    else:
                        eta_str = "N/A"
                    
                    # Handle division by zero for progress
                    if self.total_steps > 0:
                        progress_pct = (current_step / self.total_steps) * 100
                        print(f"📈 Step {current_step}/{self.total_steps} ({progress_pct:.1f}%) | Loss: {loss:.4f} | LR: {lr:.2e} | ETA: {eta_str}")
                    else:
                        print(f"📈 Step {current_step} | Loss: {loss:.4f} | LR: {lr:.2e}")
                    
            def on_train_end(self, args, state, control, **kwargs):
                import time
                if self.start_time:
                    total_time = time.time() - self.start_time
                    print(f"⏱️ Training completed in {total_time/60:.1f} minutes")

        # Calculate total training steps
        effective_batch_size = batch_size * training_args.gradient_accumulation_steps
        steps_per_epoch = max(1, len(train_dataset) // effective_batch_size)
        total_steps = max(1, steps_per_epoch * num_epochs)
        
        # Create trainer with progress callback
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=[ProgressCallback(total_steps)],
        )

        # Ensure model is in training mode
        self.model.train()

        # Train with optional resume
        if resume_from_checkpoint:
            print(f"🔄 Resuming training from checkpoint: {resume_from_checkpoint}")
            print(f"🚀 Starting training for {num_epochs} epochs...")
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            print(f"🚀 Starting training for {num_epochs} epochs...")
            trainer.train()

        # Save model
        print(f"💾 Saving LoRA model to {self.output_dir}")
        trainer.save_model()

        print("✅ Training completed!")

    def save_for_inference(self, save_path: str = "./sf2_lora_inference"):
        """Save model for inference"""
        self.model.save_pretrained(save_path)
        print(f"💾 Model saved for inference at {save_path}")


def collect_gameplay_data(
    num_episodes: int = 5, existing_agent=None
) -> StreetFighterDataset:
    """Collect training data by playing the game"""
    import numpy as np
    from PIL import Image
    import cv2
    import gymnasium as gym
    from gymnasium import spaces
    import retro
    from discretizer import StreetFighter2Discretizer

    # Simple StreetFighter wrapper for data collection
    class StreetFighter(gym.Env):
        def __init__(self):
            super().__init__()
            # Create retro environment
            game = retro.make(
                "StreetFighterIISpecialChampionEdition-Genesis",
                state="ken_bison_12.state",
                use_restricted_actions=retro.Actions.FILTERED,
            )
            self.game = StreetFighter2Discretizer(game)
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
            )
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

            # Health tracking and reward calculation
            current_agent_hp = info.get("agent_hp", self.agent_hp)
            current_enemy_hp = info.get("enemy_hp", self.enemy_hp)

            delta_hp_diff = (current_agent_hp - self.agent_hp) - (
                current_enemy_hp - self.enemy_hp
            )
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

    print(f"🎮 Collecting gameplay data for {num_episodes} episodes...")

    # Use the same wrapper as the main training
    env = StreetFighter()

    # Use existing agent if provided, otherwise create mock data
    if existing_agent:
        agent = existing_agent
    else:
        # Create expert rule-based agent for better training data
        class ExpertAgent:
            def __init__(self):
                self.frame_counter = 0
                self.last_action = 0
                self.action_repeat_count = 0
                # Quality filtering state
                self.last_sampled_step = -100
                self.recent_game_states = []
                self.action_distribution = {}
                self.last_hp_values = {"agent": 176, "enemy": 176}

            def reset(self):
                self.frame_counter = 0
                self.last_action = 0
                self.action_repeat_count = 0
                # Reset quality filtering state
                self.last_sampled_step = -100
                self.recent_game_states = []
                self.action_distribution = {}
                self.last_hp_values = {"agent": 176, "enemy": 176}

            def capture_game_frame(self, obs):
                from PIL import Image
                import cv2

                # Get raw observation from retro environment instead of preprocessed one
                try:
                    raw_obs = env.game.env.unwrapped.render()  # Get raw RGB frame
                    if raw_obs is not None and isinstance(raw_obs, np.ndarray):
                        if len(raw_obs.shape) == 3 and raw_obs.shape[2] == 3:
                            # Resize to model's expected input size
                            resized = cv2.resize(
                                raw_obs, (320, 224), interpolation=cv2.INTER_CUBIC
                            )
                            return Image.fromarray(resized)
                except:
                    pass

                # Fallback: try to use the preprocessed obs if it has visual info
                if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
                    if obs.shape[2] == 1:  # Grayscale
                        # Convert back to RGB
                        rgb_obs = np.repeat(obs, 3, axis=2)
                        return Image.fromarray(rgb_obs.astype(np.uint8))
                    elif obs.shape[2] == 3:  # RGB
                        return Image.fromarray(obs.astype(np.uint8))

                # Last resort: create a meaningful placeholder
                return Image.fromarray(np.ones((224, 320, 3), dtype=np.uint8) * 128)

            def extract_game_features(self, info):
                return info

            def get_action(self, obs, info, verbose=False):
                """Anti-Bison expert agent based on SF2 boss strategies"""
                _ = obs  # Mark as used

                # Extract game state
                agent_hp = info.get("agent_hp", 176)
                enemy_hp = info.get("enemy_hp", 176)
                agent_x = info.get("agent_x", 100)
                agent_y = info.get("agent_y", 200)
                enemy_x = info.get("enemy_x", 200)
                enemy_y = info.get("enemy_y", 200)
                enemy_status = info.get("enemy_status", 0)
                distance = abs(agent_x - enemy_x)

                # Calculate advantage
                hp_advantage = agent_hp - enemy_hp
                facing_right = agent_x < enemy_x

                # Generate enhanced reasoning with detailed analysis
                action = 0  # Default NO_ACTION
                reasoning = self._generate_enhanced_reasoning(
                    agent_hp,
                    enemy_hp,
                    agent_x,
                    agent_y,
                    enemy_x,
                    enemy_y,
                    enemy_status,
                    distance,
                    hp_advantage,
                    facing_right,
                )

                # Anti-Bison Strategy based on research

                # Priority 1: Bison's vulnerable moments
                if enemy_status != 0:  # Enemy in recovery/animation
                    if distance < 50:
                        action = 17  # HEAVY_PUNCH (punish recovery)
                    else:
                        action = 6 if facing_right else 3  # Move closer to punish

                # Priority 2: Counter Bison's range advantage
                elif distance > 100:  # Far range - Bison loves Psycho Crusher
                    if self.frame_counter % 3 == 0:
                        action = 2  # DOWN (crouch to avoid Psycho Crusher)
                    elif self.frame_counter % 3 == 1:
                        action = 38 if facing_right else 41  # Counter with projectile
                    else:
                        action = 6 if facing_right else 3  # Careful approach

                # Priority 3: Mid-range poking (Bison's weak spot)
                elif distance > 60 and distance <= 100:  # Medium range
                    frame_mod = self.frame_counter % 8  # More variety
                    if frame_mod == 0:
                        action = 26  # MEDIUM_KICK (outrange Bison normals)
                    elif frame_mod == 1:
                        action = 13  # MEDIUM_PUNCH (safe poke)
                    elif frame_mod == 2:
                        action = 21  # LIGHT_KICK (quick poke)
                    elif frame_mod == 3:
                        action = 9  # LIGHT_PUNCH (quick attack)
                    elif frame_mod == 4:
                        action = 32  # HEAVY_KICK (power attack)
                    elif frame_mod == 5:
                        action = 38 if facing_right else 41  # HADOKEN
                    elif frame_mod == 6:
                        action = 2  # DOWN (defensive)
                    else:
                        action = 1  # UP (jump over potential Head Stomp)

                # Priority 4: Close range - very dangerous vs Bison
                elif distance <= 60:  # Close range - Bison is deadly here
                    if hp_advantage > 20:  # Only aggressive if winning
                        if self.frame_counter % 3 == 0:
                            action = 9  # LIGHT_PUNCH (safest attack)
                        elif self.frame_counter % 3 == 1:
                            action = 21  # LIGHT_KICK (different timing)
                        else:
                            action = 2  # DOWN (block expected counter)
                    else:  # Losing or even - be very careful
                        if self.frame_counter % 5 == 0:
                            action = 9  # LIGHT_PUNCH (quick poke)
                        elif self.frame_counter % 5 == 1:
                            action = 2  # DOWN (block)
                        elif self.frame_counter % 5 == 2:
                            action = 3 if facing_right else 6  # Back away
                        elif self.frame_counter % 5 == 3:
                            action = 1  # UP (jump away)
                        else:
                            action = 0  # NO_ACTION (wait for opening)

                # Health-based modifiers
                if hp_advantage < -50:  # Desperately losing
                    if distance < 40 and self.frame_counter % 6 == 0:
                        action = 17  # HEAVY_PUNCH (desperate damage)
                    elif distance > 80:
                        action = 38 if facing_right else 41  # HADOKEN (chip damage)

                # Prevent action repetition (causes predictable blocking)
                if action == self.last_action:
                    self.action_repeat_count += 1
                    if self.action_repeat_count > 2:  # Don't repeat more than 2 times
                        # Choose alternative action
                        if action in [9, 13, 17]:  # If punching, try kicking
                            action = 21  # LIGHT_KICK
                        elif action in [21, 26, 32]:  # If kicking, try punching
                            action = 9  # LIGHT_PUNCH
                        elif action in [3, 6]:  # If moving, try jumping
                            action = 1  # UP
                        else:  # Default to no action
                            action = 0  # NO_ACTION
                        self.action_repeat_count = 0
                else:
                    self.action_repeat_count = 0

                # Update reasoning with final action choice and strategic context
                reasoning = self._finalize_reasoning(
                    reasoning, action, hp_advantage, distance, enemy_status
                )

                self.last_action = action
                self.frame_counter += 1

                if verbose:
                    print(
                        f"🧠 Expert: HP {agent_hp} vs {enemy_hp} (diff: {hp_advantage:+d}), Dist: {distance}, Action: {action}"
                    )

                return action, reasoning

            def _generate_enhanced_reasoning(
                self,
                agent_hp,
                enemy_hp,
                agent_x,
                agent_y,
                enemy_x,
                enemy_y,
                enemy_status,
                distance,
                hp_advantage,
                facing_right,
            ):
                """Generate detailed reasoning with state analysis"""

                # State analysis
                health_status = self._analyze_health_situation(
                    agent_hp, enemy_hp, hp_advantage
                )
                position_analysis = self._analyze_positioning(
                    distance, agent_x, enemy_x, facing_right
                )
                enemy_state = self._analyze_enemy_state(enemy_status, distance)

                reasoning = (
                    f"SITUATION: {health_status} {position_analysis} {enemy_state}"
                )
                return reasoning

            def _analyze_health_situation(self, agent_hp, enemy_hp, hp_advantage):
                """Analyze current health situation"""
                agent_hp_pct = (agent_hp / 176.0) * 100
                enemy_hp_pct = (enemy_hp / 176.0) * 100

                if hp_advantage > 50:
                    return f"Winning decisively ({agent_hp_pct:.0f}% vs {enemy_hp_pct:.0f}%)."
                elif hp_advantage > 20:
                    return (
                        f"Ahead in health ({agent_hp_pct:.0f}% vs {enemy_hp_pct:.0f}%)."
                    )
                elif hp_advantage > -20:
                    return f"Even match ({agent_hp_pct:.0f}% vs {enemy_hp_pct:.0f}%)."
                elif hp_advantage > -50:
                    return f"Behind in health ({agent_hp_pct:.0f}% vs {enemy_hp_pct:.0f}%)."
                else:
                    return f"Desperately losing ({agent_hp_pct:.0f}% vs {enemy_hp_pct:.0f}%)."

            def _analyze_positioning(self, distance, agent_x, enemy_x, facing_right):
                """Analyze current positioning and spacing"""
                if distance > 100:
                    return f"Bison at long range ({distance}px) - danger of Psycho Crusher/Scissor Kick."
                elif distance > 60:
                    return f"Mid-range ({distance}px) - optimal footsies distance vs Bison."
                elif distance > 30:
                    return (
                        f"Close range ({distance}px) - Bison's deadly zone, high risk."
                    )
                else:
                    return f"Point-blank ({distance}px) - extremely dangerous vs Bison's normals."

            def _analyze_enemy_state(self, enemy_status, distance):
                """Analyze enemy's current state and vulnerabilities"""
                if enemy_status != 0:
                    if distance < 50:
                        return "Bison in recovery - punish opportunity!"
                    else:
                        return "Bison vulnerable - close distance for punish."
                else:
                    return "Bison active - expect charge moves or normals."

            def _finalize_reasoning(
                self, base_reasoning, action, hp_advantage, distance, enemy_status
            ):
                """Add action justification and strategic context"""
                action_explanation = self._explain_action(
                    action, hp_advantage, distance, enemy_status
                )
                strategic_context = self._add_strategic_context(
                    action, hp_advantage, distance
                )

                return f"{base_reasoning} ACTION: {action_explanation} STRATEGY: {strategic_context}"

            def _explain_action(self, action, hp_advantage, distance, enemy_status):
                """Explain why this specific action was chosen"""
                action_names = {
                    0: "No action",
                    1: "Jump",
                    2: "Crouch",
                    3: "Move back",
                    6: "Move forward",
                    9: "Light punch",
                    13: "Medium punch",
                    17: "Heavy punch",
                    21: "Light kick",
                    26: "Medium kick",
                    32: "Heavy kick",
                    38: "Hadoken right",
                    41: "Hadoken left",
                }

                action_name = action_names.get(action, f"Action {action}")

                if enemy_status != 0 and action == 17:
                    return f"{action_name} to punish Bison's recovery frames."
                elif distance > 100 and action == 2:
                    return f"{action_name} to avoid Bison's charge attacks."
                elif distance > 100 and action in [38, 41]:
                    return f"{action_name} to counter-zone Bison's long-range game."
                elif 60 < distance <= 100 and action in [21, 26, 9, 13]:
                    return f"{action_name} - safe poke in Bison's weak mid-range."
                elif distance <= 60 and hp_advantage > 20 and action in [9, 21]:
                    return f"{action_name} - aggressive but safe since ahead."
                elif distance <= 60 and hp_advantage <= 20 and action == 2:
                    return f"{action_name} - defensive vs dangerous Bison."
                elif hp_advantage < -50 and action == 17:
                    return f"{action_name} - desperate damage attempt."
                else:
                    return f"{action_name} based on current situation."

            def _add_strategic_context(self, action, hp_advantage, distance):
                """Add strategic context and improvement suggestions"""
                if action in [9, 13, 17, 21, 26, 32]:  # Attack actions
                    if distance > 60:
                        return (
                            "Good spacing control. Follow up with movement or fireball."
                        )
                    else:
                        return "Risky close-range attack. Be ready to block Bison's counter."
                elif action in [38, 41]:  # Projectiles
                    return "Fireball controls space. Watch for Bison's jump or Scissor Kick."
                elif action == 2:  # Crouch/block
                    return "Defensive choice. Look for openings after blocking Bison's attack."
                elif action in [3, 6]:  # Movement
                    return "Repositioning vs Bison. Maintain optimal mid-range spacing."
                elif action == 1:  # Jump
                    return "Jump can avoid ground attacks but vulnerable to anti-airs."
                else:
                    return "Neutral option. Observe Bison's patterns for next move."

            def should_capture_frame(self, step, info, action, obs):
                """Determine if current frame should be captured for training"""

                # 1. Minimum time gap (don't sample too frequently)
                if step - self.last_sampled_step < 15:  # At least 15 frames apart
                    return False

                should_capture = False

                # 2. Important moment detection
                agent_hp = info.get("agent_hp", 176)
                enemy_hp = info.get("enemy_hp", 176)

                # Health change detected (important moment)
                if (
                    agent_hp != self.last_hp_values["agent"]
                    or enemy_hp != self.last_hp_values["enemy"]
                ):
                    self.last_hp_values = {"agent": agent_hp, "enemy": enemy_hp}
                    should_capture = True

                # 3. Action change (new strategy)
                if action != self.last_action:
                    should_capture = True

                # 4. Periodic sampling (but less frequent)
                if step % 60 == 0:  # Every 60 frames instead of 30
                    should_capture = True

                # 5. Critical game states
                distance = abs(info.get("agent_x", 100) - info.get("enemy_x", 200))
                if distance < 40 or distance > 150:  # Close combat or long range
                    should_capture = True

                # 6. Enemy in vulnerable state
                if info.get("enemy_status", 0) != 0:
                    should_capture = True

                if should_capture:
                    self.last_sampled_step = step

                return should_capture

            def is_quality_sample(self, image, game_state, action, reasoning):
                """Check if this sample meets quality standards"""

                # 1. No duplicate actions (simple diversity check)
                action_count = self.action_distribution.get(action, 0)
                total_samples = sum(self.action_distribution.values())

                if total_samples > 20:  # After collecting more samples
                    action_ratio = action_count / total_samples
                    if action_ratio > 0.3:  # Action appears in >30% of samples
                        return False
                elif total_samples > 5:  # Early samples - less restrictive
                    if (
                        action_count > 2
                    ):  # Don't allow more than 2 of same action early on
                        return False

                # 2. Avoid repetitive game states
                current_state_hash = self._hash_game_state(game_state)
                if current_state_hash in self.recent_game_states:
                    return False

                # 3. Quality checks
                if (
                    action == 0 and "No action" in reasoning
                ):  # Avoid too many idle frames
                    if total_samples > 0:
                        no_action_count = self.action_distribution.get(0, 0)
                        if no_action_count / total_samples > 0.2:  # >20% no-action
                            return False

                # 4. Ensure we have meaningful reasoning
                if len(reasoning) < 50:  # Too short reasoning
                    return False

                # Update tracking
                self.action_distribution[action] = action_count + 1
                self.recent_game_states.append(current_state_hash)
                if len(self.recent_game_states) > 20:  # Keep last 20 states
                    self.recent_game_states.pop(0)

                return True

            def _hash_game_state(self, game_state):
                """Create a hash of game state for diversity checking"""
                # Round values to reduce sensitivity to small changes
                agent_hp = (
                    round(game_state.get("agent_hp", 176) / 30) * 30
                )  # Less sensitive
                enemy_hp = round(game_state.get("enemy_hp", 176) / 30) * 30
                distance = (
                    round(game_state.get("distance", 100) / 50) * 50
                )  # Less sensitive

                return f"{agent_hp}_{enemy_hp}_{distance}_{game_state.get('enemy_status', 0)}"

        agent = ExpertAgent()

    dataset = StreetFighterDataset()

    # Health tracking
    agent_hp = 176
    enemy_hp = 176

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        obs = env.reset()
        agent.reset()

        # Reset health tracking
        agent_hp = 176
        enemy_hp = 176

        step = 0
        max_steps = 4000  # Longer fights for natural progression
        episode_reward = 0
        print(f"  Starting fight: Agent {agent_hp} HP vs Enemy {enemy_hp} HP")
        while step < max_steps:
            # Get action from agent using current game state
            # Extract real game state from environment
            try:
                # Try to get real game state from retro environment
                game_info = env.game.get_info()
                current_agent_hp = game_info.get("health", agent_hp)
                current_enemy_hp = game_info.get("enemy_health", enemy_hp)
                agent_x = game_info.get("x", 100 + (step % 50))  # Add some variation
                enemy_x = game_info.get("enemy_x", 200 - (step % 30))
            except:
                # Fallback with realistic fight progression
                import random

                # Gradual HP loss for longer, more realistic fights
                if step > 100:  # Start taking damage after step 100 (more buildup)
                    damage_rate = (step - 100) // 50  # Slower damage increase
                    if step % 40 == 0:  # Damage every 40 steps (less frequent)
                        agent_damage = random.randint(8, 18) + damage_rate * 3
                        enemy_damage = random.randint(10, 22) + damage_rate * 2
                        agent_hp = max(0, agent_hp - agent_damage)
                        enemy_hp = max(0, enemy_hp - enemy_damage)
                        print(
                            f"    Step {step}: Agent {current_agent_hp}→{agent_hp}, Enemy {current_enemy_hp}→{enemy_hp}"
                        )

                current_agent_hp = agent_hp
                current_enemy_hp = enemy_hp
                agent_x = 100 + (step % 100) - 50  # Agent moves around
                enemy_x = 200 + ((step * 2) % 80) - 40  # Enemy moves differently

            info = {
                "agent_hp": current_agent_hp,
                "enemy_hp": current_enemy_hp,
                "agent_x": agent_x,
                "agent_y": 200 + (step % 20) - 10,  # Some vertical movement
                "enemy_x": enemy_x,
                "enemy_y": 200 + ((step * 3) % 30) - 15,
                "distance": abs(agent_x - enemy_x),
                "enemy_status": 1 if step % 60 < 5 else 0,  # Enemy vulnerable sometimes
                "agent_won": current_enemy_hp <= 0,
            }

            # Get action from agent
            action, reasoning = agent.get_action(obs, info, verbose=False)

            # Capture current state for training with quality filtering
            if agent.should_capture_frame(step, info, action, obs):
                image = agent.capture_game_frame(obs)
                game_state = agent.extract_game_features(info)

                # Add to dataset with quality check
                if agent.is_quality_sample(image, game_state, action, reasoning):
                    dataset.add_example(image, game_state, action, reasoning)

            # Take step using wrapper (returns 5 values: obs, reward, done, truncated, info)
            obs, reward, done, truncated, step_info = env.step(action)

            # Update health tracking from actual game results
            if "agent_hp" in step_info:
                agent_hp = step_info["agent_hp"]
            if "enemy_hp" in step_info:
                enemy_hp = step_info["enemy_hp"]

            # Calculate health change (agent_hp_change - enemy_hp_change)
            # we want to create health advantage compared your opponent
            delta_hp_diff = (current_agent_hp - agent_hp) - (
                current_enemy_hp - enemy_hp
            )

            # Normalize health reward by max health (176) and add small time penalty
            # health double diff
            reward = delta_hp_diff / 176.0 - 0.0001

            # Update health tracking
            agent_hp = current_agent_hp
            enemy_hp = current_enemy_hp

            # Check for round end
            if current_agent_hp <= 0 or current_enemy_hp <= 0:
                done = True
                if current_enemy_hp <= 0 and current_agent_hp > 0:
                    reward += 1.0  # Large win bonus
                    info["agent_won"] = True
                    print(
                        f"🏆 Agent won! Final HP: {current_agent_hp} vs {current_enemy_hp}"
                    )
                else:
                    reward -= 1.0  # Large loss penalty
                    info["agent_won"] = False
                    print(
                        f"💀 Agent lost! Final HP: {current_agent_hp} vs {current_enemy_hp}"
                    )
            elif step >= max_steps:
                done = True
                print(
                    f"⏰ Time limit reached! Final HP: {current_agent_hp} vs {current_enemy_hp}"
                )

            episode_reward += reward
            step += 1

            if done:
                agent_won = info.get("agent_won", False)
                print(
                    f"Episode {episode + 1} ended: Reward = {episode_reward:.3f}, Agent Won = {agent_won}"
                )

                print(
                    f"Episode {episode + 1} ended: Reward = {episode_reward:.3f}, Agent Won = {agent_won}"
                )
                break

    env.close()
    print(f"📊 Collected {len(dataset.data)} training examples")
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning for Street Fighter Qwen Agent"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-3B-Instruct-AWQ",
        help="Base model path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./sf2_lora_model",
        help="Output directory for LoRA model",
    )
    parser.add_argument(
        "--data-path", type=str, default=None, help="Path to training data JSON file"
    )
    parser.add_argument(
        "--collect-data",
        action="store_true",
        help="Collect training data by playing the game",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to collect data from",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument(
        "--append-data",
        action="store_true",
        help="Append new data to existing dataset instead of overwriting",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint directory",
    )
    parser.add_argument(
        "--save-steps", type=int, default=100, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="Only collect data, skip training"
    )
    parser.add_argument(
        "--no-shuffle", action="store_true", help="Don't shuffle training data (use sequential order)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None, help="Randomly sample this many examples for training"
    )
    # CPU offload removed - AWQ models don't support it

    args = parser.parse_args()

    # Initialize trainer
    trainer = StreetFighterLoRATrainer(
        model_path=args.model,  # Always use base model path
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    # Collect or load data
    if args.collect_data:
        print("🧠 Using expert rule-based agent for better training data")
        dataset = collect_gameplay_data(
            args.episodes, existing_agent=None
        )  # Use mock agent
        # Save collected data using save_data method with append option
        dataset.save_data("./data/sf2_training_data.json", append=args.append_data)

        if args.no_train:
            print(
                "✅ Data collection completed! Training skipped due to --no-train flag."
            )
            return
    else:
        dataset = StreetFighterDataset(
            args.data_path, 
            shuffle=not args.no_shuffle,
            sample_size=args.sample_size
        )
        if len(dataset.data) == 0:
            print(
                "❌ No training data found. Use --collect-data to collect data first."
            )
            return

    # Train model (only if not skipped)
    if not args.no_train:
        trainer.train(
            dataset=dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_steps=args.save_steps,
            resume_from_checkpoint=args.resume_from,
        )

        # No need for separate inference directory - use checkpoints directly
    else:
        print("⏭️ Training skipped due to --no-train flag.")


if __name__ == "__main__":
    main()
