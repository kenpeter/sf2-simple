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
    
    def __init__(self, data_path: str = None):
        self.data = []
        if data_path is not None and os.path.exists(data_path):
            self.load_data(data_path)
    
    def load_data(self, data_path: str):
        """Load training data from JSON file with base64 image decoding"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Convert base64 images back to PIL Images
        self.data = []
        for item in data:
            if isinstance(item['image'], str):  # base64 encoded
                # Decode base64 image
                image_data = base64.b64decode(item['image'])
                image = Image.open(BytesIO(image_data))
                item['image'] = image
            self.data.append(item)
        
        print(f"📁 Loaded {len(self.data)} training examples")
    
    def add_example(self, image: Image.Image, game_state: Dict, action: int, reasoning: str):
        """Add a training example"""
        example = {
            'image': image,  # Keep as PIL Image for processing
            'game_state': game_state,
            'action': action,
            'reasoning': reasoning
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
                with open(data_path, 'r') as f:
                    existing_data = json.load(f)
                print(f"📁 Found {len(existing_data)} existing training examples")
            except (json.JSONDecodeError, FileNotFoundError):
                print("⚠️ Could not load existing data, creating new file")
                existing_data = []
        
        # Convert current data to JSON-serializable format
        serializable_data = []
        for item in self.data:
            # Convert PIL Image to base64
            if isinstance(item['image'], Image.Image):
                buffered = BytesIO()
                item['image'].save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_data = img_base64
            else:
                image_data = item['image']  # Already encoded
            
            serializable_item = {
                'image': image_data,
                'game_state': item['game_state'],
                'action': item['action'],
                'reasoning': item['reasoning']
            }
            serializable_data.append(serializable_item)
        
        # Combine with existing data if appending
        if append:
            all_data = existing_data + serializable_data
            print(f"📊 Appending {len(serializable_data)} new examples to {len(existing_data)} existing ones")
        else:
            all_data = serializable_data
        
        # Save to JSON file
        with open(data_path, 'w') as f:
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
        model_path: str = "/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-7B-Instruct-AWQ",
        output_dir: str = "./sf2_lora_model",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        
        # Initialize model and processor
        print(f"🤖 Loading base model: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        
        # Fix tokenizer if needed
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        
        # Load model with memory optimization
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",  # Use auto device mapping for CPU offloading
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "10GB", "cpu": "30GB"}  # Limit GPU to 10GB, use CPU for overflow
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
            prompt = dataset.create_prompt(example['game_state'])
            
            # Create full conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example['image']},
                        {"type": "text", "text": prompt}
                    ]
                },
                {
                    "role": "assistant", 
                    "content": str(example['action'])
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            
            # Process inputs using the same method as qwen_agent.py
            # Convert PIL image to format expected by processor
            image = example['image']
            if hasattr(image, 'convert'):
                image = image.convert('RGB')
            
            # Use the processor with proper image handling
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            result = {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': inputs['input_ids'].squeeze(),  # For causal LM, labels = input_ids
            }
            
            # Add all necessary image-related tensors
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                result['pixel_values'] = inputs['pixel_values'].squeeze()
            
            if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
                result['image_grid_thw'] = inputs['image_grid_thw'].squeeze()
            
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
    ):
        """Train the LoRA model"""
        # Prepare dataset
        train_dataset = self.prepare_dataset(dataset)
        
        # Training arguments - optimized for GPU memory
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,  # Increase to maintain effective batch size
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=save_steps,
            save_total_limit=2,  # Reduce saved checkpoints
            warmup_steps=20,  # Reduce warmup steps
            fp16=True,
            gradient_checkpointing=False,  # Disable - causes issues with LoRA
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Reduce memory overhead
            optim="adamw_torch",  # Use more memory efficient optimizer
            deepspeed="ds_config.json" if os.path.exists("ds_config.json") else None,
        )
        
        # Custom data collator with proper tensor handling
        def data_collator(batch):
            # Convert all tensors and ensure proper dtypes
            input_ids = torch.stack([
                torch.tensor(item['input_ids'], dtype=torch.long) if not isinstance(item['input_ids'], torch.Tensor)
                else item['input_ids'].long() for item in batch
            ])
            attention_mask = torch.stack([
                torch.tensor(item['attention_mask'], dtype=torch.long) if not isinstance(item['attention_mask'], torch.Tensor)
                else item['attention_mask'].long() for item in batch
            ])
            labels = torch.stack([
                torch.tensor(item['labels'], dtype=torch.long) if not isinstance(item['labels'], torch.Tensor)
                else item['labels'].long() for item in batch
            ])
            
            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
            
            # Add pixel values if present
            if 'pixel_values' in batch[0] and batch[0]['pixel_values'] is not None:
                pixel_values = torch.stack([
                    torch.tensor(item['pixel_values'], dtype=torch.float16) if not isinstance(item['pixel_values'], torch.Tensor)
                    else item['pixel_values'].half() for item in batch
                ])
                result['pixel_values'] = pixel_values
            
            # Add image grid thw if present
            if 'image_grid_thw' in batch[0] and batch[0]['image_grid_thw'] is not None:
                image_grid_thw = torch.stack([
                    torch.tensor(item['image_grid_thw'], dtype=torch.long) if not isinstance(item['image_grid_thw'], torch.Tensor)
                    else item['image_grid_thw'].long() for item in batch
                ])
                result['image_grid_thw'] = image_grid_thw
            
            return result
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Ensure model is in training mode
        self.model.train()
        
        # Train
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


def collect_gameplay_data(num_episodes: int = 5, existing_agent=None) -> StreetFighterDataset:
    """Collect training data by playing the game"""
    import numpy as np
    from PIL import Image
    from wrapper import StreetFighter
    
    print(f"🎮 Collecting gameplay data for {num_episodes} episodes...")
    
    # Use the same wrapper as the main training
    env = StreetFighter()
    
    # Use existing agent if provided, otherwise create mock data
    if existing_agent:
        agent = existing_agent
    else:
        # Create mock agent for data collection without loading model
        class MockAgent:
            def __init__(self):
                self.frame_counter = 0
                
            def reset(self):
                self.frame_counter = 0
                
            def capture_game_frame(self, obs):
                if isinstance(obs, np.ndarray):
                    if obs.dtype != np.uint8:
                        obs = (obs * 255).astype(np.uint8)
                    if len(obs.shape) == 3 and obs.shape[2] in [3, 4]:
                        if obs.shape[2] == 4:
                            obs = obs[:, :, :3]
                        return Image.fromarray(obs)
                return Image.fromarray(np.zeros((224, 320, 3), dtype=np.uint8))
                
            def extract_game_features(self, info):
                return info
                
            def get_action(self, obs, info, verbose=False):
                # Random action selection for data collection  
                _ = obs, info, verbose  # Mark as used
                action = random.randint(0, 43)
                reasoning = f"Random action {action} for data collection"
                return action, reasoning
                
        agent = MockAgent()
        
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
        max_steps = 1000
        episode_reward = 0
        
        while step < max_steps:
            # Create info with health tracking (mock values for now)
            current_agent_hp = agent_hp  # Would extract from game memory in real implementation
            current_enemy_hp = enemy_hp  # Would extract from game memory in real implementation
            
            info = {
                'agent_hp': current_agent_hp,
                'enemy_hp': current_enemy_hp,
                'agent_x': 100,
                'agent_y': 200,
                'enemy_x': 200,
                'enemy_y': 200,
                'distance': 100,
                'enemy_status': 0,
                'agent_won': False
            }
            
            # Get action from agent
            action, reasoning = agent.get_action(obs, info, verbose=False)
            
            # Capture current state for training
            if step % 30 == 0:  # Sample every 30 frames
                image = agent.capture_game_frame(obs)
                game_state = agent.extract_game_features(info)
                
                # Add to dataset
                dataset.add_example(image, game_state, action, reasoning)
            
            # Take step using wrapper (returns 5 values: obs, reward, done, truncated, info)
            obs, _, done, _, _ = env.step(action)
            
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
            if agent_hp <= 0 or enemy_hp <= 0:
                done = True
                if enemy_hp <= 0 and agent_hp > 0:
                    reward += 1.0  # Large win bonus
                    info["agent_won"] = True
                else:
                    reward -= 1.0  # Large loss penalty
                    info["agent_won"] = False
            
            episode_reward += reward
            step += 1
            
            if done:
                print(f"Episode {episode + 1} ended: Reward = {episode_reward:.3f}, Agent Won = {info.get('agent_won', False)}")
                break
    
    env.close()
    print(f"📊 Collected {len(dataset.data)} training examples")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Street Fighter Qwen Agent")
    parser.add_argument("--model", type=str, 
                       default="/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-7B-Instruct-AWQ",
                       help="Base model path")
    parser.add_argument("--output-dir", type=str, default="./sf2_lora_model",
                       help="Output directory for LoRA model")
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to training data JSON file")
    parser.add_argument("--collect-data", action="store_true",
                       help="Collect training data by playing the game")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to collect data from")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--append-data", action="store_true",
                       help="Append new data to existing dataset instead of overwriting")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume training from checkpoint directory")
    parser.add_argument("--save-steps", type=int, default=100,
                       help="Save checkpoint every N steps")
    parser.add_argument("--no-train", action="store_true",
                       help="Only collect data, skip training")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = StreetFighterLoRATrainer(
        model_path=args.resume_from if args.resume_from else args.model,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    
    # Collect or load data
    if args.collect_data:
        print("📊 Using mock agent for data collection to save GPU memory")
        dataset = collect_gameplay_data(args.episodes, existing_agent=None)  # Use mock agent
        # Save collected data using save_data method with append option
        dataset.save_data("./data/sf2_training_data.json", append=args.append_data)
    else:
        dataset = StreetFighterDataset(args.data_path)
        if len(dataset.data) == 0:
            print("❌ No training data found. Use --collect-data to collect data first.")
            return
    
    # Train model
    trainer.train(
        dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
    )
    
    # Save for inference
    trainer.save_for_inference()


if __name__ == "__main__":
    main()