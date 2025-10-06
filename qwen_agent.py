#!/usr/bin/env python3  # Shebang line to run script with python3 directly
"""
🥊 Qwen-powered Street Fighter 2 Agent
Works with existing wrapper.py without modifications
Includes demo functionality for testing and gameplay
"""

# Import PyTorch for deep learning functionality
import torch

# Import HuggingFace transformers for vision models
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from peft import PeftModel  # Import PEFT for LoRA adapter loading

# Import NumPy for numerical array operations
import numpy as np

# for img
from PIL import Image
import re  # Import regular expressions for text pattern matching
from typing import Dict, Tuple  # Import typing hints for better code documentation


class QwenStreetFighterAgent:  # Define main agent class for Street Fighter 2 AI
    """
    Qwen-powered agent for Street Fighter 2
    Uses existing wrapper.py environment without modifications
    """

    # agent init
    def __init__(
        self,
        fresh_start: bool = False,  # If True, copy from cache to current dir
    ):  # Constructor method for agent initialization
        """
        Initialize the Qwen agent

        Args:
            fresh_start: If True, copy fresh model from cache to current dir
        """
        # Setup model paths
        self.cache_model_path = "/home/kenpeter/.cache/huggingface/hub/Qwen2-VL-2B-Instruct"
        self.local_model_path = "./qwen_model"
        
        # Setup model path based on fresh_start
        if fresh_start:
            self.setup_fresh_model()
            model_path = self.local_model_path
        else:
            # Resume: use local model if exists, otherwise copy from cache
            if self.model_exists_locally():
                model_path = self.local_model_path
                print("📁 Resuming from local model")
            else:
                print("🆕 No local model found, copying from cache")
                self.setup_fresh_model()
                model_path = self.local_model_path

        # Initialize Qwen model
        print(f"🤖 Loading Qwen 2B model from: {model_path}")

        # device cuda
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Set device to GPU if available, else CPU

        # Multi-frame context for temporal understanding - 8 frame stack
        self.frame_history = []  # Store recent frames for temporal context
        self.max_history_frames = 8  # Keep last 8 frames for context (frame stacking)

        # Load processor and model for vision
        print(
            "📁 Step 1/2: Loading processor from cache..."
        )  # Print loading status for processor

        # so this has tokenizer and image processor
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )  # Load tokenizer and image processor

        # Fix tokenizer configuration for CUDA compatibility
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # Load vision model from cache with INT8 quantization
        print(
            "📁 Step 2/2: Loading Qwen2-VL model from cache..."
        )  # Print loading status for model

        # Load 2B model with fp16 for GPU efficiency
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda:0",  # GPU for inference
            torch_dtype=torch.float16,  # fp16 for efficiency
            local_files_only=True,
            trust_remote_code=True,
        )
        print(
            f"✅ Qwen 2B model loaded successfully on {self.device}"
        )  # Print successful loading message

        #
        self.action_meanings = [  # Define all possible actions the agent can take
            "NO_ACTION",  # 0 - Do nothing action
            "UP",  # 1 - Jump upward
            "DOWN",  # 2 - Crouch downward
            "LEFT",  # 3 - Move left
            "UP_LEFT",  # 4 - Jump diagonally left
            "DOWN_LEFT",  # 5 - Crouch walk left
            "RIGHT",  # 6 - Move right
            "UP_RIGHT",  # 7 - Jump diagonally right
            "DOWN_RIGHT",  # 8 - Crouch walk right
            "LIGHT_PUNCH",  # 9 - Quick punch attack
            "LIGHT_PUNCH_DOWN",  # 10 - Crouching light punch
            "LIGHT_PUNCH_LEFT",  # 11 - Light punch while moving left
            "LIGHT_PUNCH_RIGHT",  # 12 - Light punch while moving right
            "MEDIUM_PUNCH",  # 13 - Medium strength punch
            "MEDIUM_PUNCH_DOWN",  # 14 - Crouching medium punch
            "MEDIUM_PUNCH_LEFT",  # 15 - Medium punch while moving left
            "MEDIUM_PUNCH_RIGHT",  # 16 - Medium punch while moving right
            "HEAVY_PUNCH",  # 17 - Strong punch attack
            "HEAVY_PUNCH_DOWN",  # 18 - Crouching heavy punch
            "HEAVY_PUNCH_LEFT",  # 19 - Heavy punch while moving left
            "HEAVY_PUNCH_RIGHT",  # 20 - Heavy punch while moving right
            "LIGHT_KICK",  # 21 - Quick kick attack
            "LIGHT_KICK_DOWN",  # 22 - Crouching light kick
            "LIGHT_KICK_LEFT",  # 23 - Light kick while moving left
            "LIGHT_KICK_DOWN_LEFT",  # 24 - Crouching light kick moving left
            "LIGHT_KICK_RIGHT",  # 25 - Light kick while moving right
            "MEDIUM_KICK",  # 26 - Medium strength kick
            "MEDIUM_KICK_DOWN",  # 27 - Crouching medium kick
            "MEDIUM_KICK_LEFT",  # 28 - Medium kick while moving left
            "MEDIUM_KICK_DOWN_LEFT",  # 29 - Crouching medium kick moving left
            "MEDIUM_KICK_RIGHT",  # 30 - Medium kick while moving right
            "MEDIUM_KICK_DOWN_RIGHT",  # 31 - Crouching medium kick moving right
            "HEAVY_KICK",  # 32 - Strong kick attack
            "HEAVY_KICK_DOWN",  # 33 - Crouching heavy kick
            "HEAVY_KICK_LEFT",  # 34 - Heavy kick while moving left
            "HEAVY_KICK_DOWN_LEFT",  # 35 - Crouching heavy kick moving left
            "HEAVY_KICK_RIGHT",  # 36 - Heavy kick while moving right
            "HEAVY_KICK_DOWN_RIGHT",  # 37 - Crouching heavy kick moving right
            "HADOKEN_RIGHT",  # 38 - Fireball special move facing right
            "DRAGON_PUNCH_RIGHT",  # 39 - Uppercut special move facing right
            "HURRICANE_KICK_RIGHT",  # 40 - Spinning kick special move facing right
            "HADOKEN_LEFT",  # 41 - Fireball special move facing left
            "DRAGON_PUNCH_LEFT",  # 42 - Uppercut special move facing left
            "HURRICANE_KICK_LEFT",  # 43 - Spinning kick special move facing left
        ]

        self.num_actions = len(
            self.action_meanings
        )  # Store total number of actions available

        # Action recovery frames based on actual SF2 Turbo frame data
        self.action_frames = {  # Dictionary mapping action IDs to total animation durations (startup+active+recovery)
            0: 1,  # NO_ACTION - instant response
            1: 15,  # UP - jump has long recovery frames (estimated)
            2: 5,  # DOWN - crouch animation frames (estimated)
            3: 3,  # LEFT - walk animation frames (estimated)
            6: 3,  # RIGHT - walk animation frames (estimated)
            7: 20,  # UP_RIGHT - jump animation frames (estimated)
            4: 20,  # UP_LEFT - jump animation frames (estimated)
            5: 8,  # DOWN_LEFT - crouch walk animation frames (estimated)
            8: 8,  # DOWN_RIGHT - crouch walk animation frames (estimated)
            9: 11,  # LIGHT_PUNCH - Jab: 2+4+5=11 frames (SF2T data)
            13: 9,  # MEDIUM_PUNCH - Strong: 1+2+6=9 frames (SF2T data)
            17: 20,  # HEAVY_PUNCH - Fierce: estimated ~20 frames (conservative)
            21: 12,  # LIGHT_KICK - Short kick: estimated similar to jab
            26: 15,  # MEDIUM_KICK - Forward kick: estimated medium timing
            32: 25,  # HEAVY_KICK - Roundhouse: estimated heavy timing
            38: 51,  # HADOKEN_RIGHT - Jab Hadouken: 10+40+1=51 frames (SF2T data)
            39: 34,  # DRAGON_PUNCH_RIGHT - Fierce Shoryuken: 4+4+26=34 frames (SF2T data)
            40: 30,  # HURRICANE_KICK_RIGHT - Roundhouse Tatsumaki: 11+3+16=30 frames (SF2T data)
            41: 51,  # HADOKEN_LEFT - Same as right hadouken
            42: 34,  # DRAGON_PUNCH_LEFT - Same as right shoryuken
            43: 30,  # HURRICANE_KICK_LEFT - Same as right tatsumaki
        }

        # Game state tracking
        self.action_history = []  # List to store history of past actions
        self.last_features = {}  # Dictionary to store previous game features
        self.frame_counter = 0  # Counter to track current frame number
        self.last_action = 0  # Store the last action taken
        self.last_reasoning = "Initial state"  # Store reasoning for last decision

        # Simple action management
        self.action_repeat_count = 0  # Count how many times same action repeated

        # action has cool down period
        self.action_cooldown = 0
        # remember last executed action
        self.last_executed_action = 0

        self.frames_since_last_action = 0

        # Add action head for head-only fine-tuning (after action_meanings is defined)
        self._setup_head_training()
        
        # Online learning setup (always enabled)
        self.online_learning = True
        self.optimizer = None
        self.criterion = None
        self.training_buffer = []
        self.buffer_size = 50  # Train every 50 samples
        
        # Auto-enable online learning
        self.enable_online_learning()
        
        # Auto-load components if resuming
        if not fresh_start and self.model_exists_locally():
            self.load_components()

    def model_exists_locally(self):
        """Check if model exists in current directory with all required files"""
        import os
        if not os.path.exists(self.local_model_path):
            return False
        
        # Check for essential files
        required_files = ['config.json', 'tokenizer_config.json']
        for file in required_files:
            if not os.path.exists(os.path.join(self.local_model_path, file)):
                return False
        return True
    
    def setup_fresh_model(self):
        """Copy fresh model from cache to current directory"""
        import os
        import shutil
        
        # Remove existing local model if present
        if os.path.exists(self.local_model_path):
            print(f"🗑️ Removing existing local model: {self.local_model_path}")
            shutil.rmtree(self.local_model_path)
        
        # Copy from cache to current directory
        print(f"📋 Copying fresh model from cache to: {self.local_model_path}")
        shutil.copytree(self.cache_model_path, self.local_model_path)
        print("✅ Fresh model setup complete")
    
    def save_model(self):
        """Save current model state to local directory"""
        import torch
        import os
        
        print(f"💾 Saving components to: {self.local_model_path}")
        
        # Create directory if needed
        os.makedirs(self.local_model_path, exist_ok=True)
        
        # Save action head
        torch.save(self.action_head.state_dict(), os.path.join(self.local_model_path, "action_head.pth"))
        
        # Save vision projector if exists
        if hasattr(self, 'vision_projector'):
            torch.save(self.vision_projector.state_dict(), os.path.join(self.local_model_path, "vision_projector.pth"))
        
        print("✅ Model components saved successfully")
    
    def load_components(self):
        """Load saved components"""
        import torch
        import os
        
        print(f"📁 Loading components from: {self.local_model_path}")
        
        # Load action head
        action_head_path = os.path.join(self.local_model_path, "action_head.pth")
        if os.path.exists(action_head_path):
            self.action_head.load_state_dict(torch.load(action_head_path, map_location=self.device))
            print("✅ Action head loaded")
        
        # Load vision projector if exists
        vision_proj_path = os.path.join(self.local_model_path, "vision_projector.pth")
        if os.path.exists(vision_proj_path):
            # Create projector first if it doesn't exist
            if not hasattr(self, 'vision_projector'):
                dummy_image = torch.zeros(1, 3, 224, 224).to(self.device)
                _ = self.get_visual_features([dummy_image])  # This creates the projector
            
            self.vision_projector.load_state_dict(torch.load(vision_proj_path, map_location=self.device))
            print("✅ Vision projector loaded")
        
        print("✅ All components loaded successfully")

    def _setup_head_training(self):
        """Setup full model fine-tuning mode"""
        import torch.nn as nn
        
        print("🔥 Setting up FULL MODEL fine-tuning mode")
        
        # Unfreeze all Qwen model parameters for full fine-tuning
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Add trainable action head
        hidden_size = getattr(self.model.config, 'hidden_size', 1280)  # Qwen2-VL-2B uses 1280
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, len(self.action_meanings))
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        action_head_params = sum(p.numel() for p in self.action_head.parameters())
        
        print(f"📊 FULL MODEL Fine-tuning Parameters:")
        print(f"  Total Qwen model: {total_params:,}")
        print(f"  Trainable Qwen: {trainable_params:,}")
        print(f"  Action head: {action_head_params:,}")
        print(f"  Total trainable: {trainable_params + action_head_params:,}")
        print(f"  Percentage trainable: {((trainable_params + action_head_params)/(total_params + action_head_params))*100:.1f}%")
    
    def get_visual_features(self, images):
        """Extract visual features using simple CNN approach"""
        try:
            # Convert PIL images to tensors manually
            import torchvision.transforms as transforms
            
            # Simple transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Process images
            if isinstance(images, list):
                tensors = [transform(img) for img in images]
                batch = torch.stack(tensors).to(self.device)
            else:
                batch = transform(images).unsqueeze(0).to(self.device)
            
            # Simple feature extraction: flatten and project
            batch_size = batch.shape[0]
            flattened = batch.view(batch_size, -1)  # Flatten to [batch, 224*224*3]
            
            # Project to hidden size
            hidden_size = getattr(self.model.config, 'hidden_size', 1280)  # Qwen2-VL-2B uses 1280
            if not hasattr(self, 'vision_projector'):
                # Create simple projection layer
                import torch.nn as nn
                input_size = flattened.shape[1]
                self.vision_projector = nn.Sequential(
                    nn.Linear(input_size, 2048),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(2048, hidden_size)
                ).to(self.device)
            
            # Project to feature space
            visual_features = self.vision_projector(flattened)
            return visual_features
            
        except Exception as e:
            print(f"⚠️ Vision feature extraction failed: {e}")
            # Fallback: random features
            batch_size = len(images) if isinstance(images, list) else 1
            hidden_size = getattr(self.model.config, 'hidden_size', 1536)
            return torch.randn(batch_size, hidden_size, device=self.device)
    
    def predict_action_direct(self, observation):
        """Direct action prediction using head-only model"""
        if not hasattr(self, 'action_head'):
            raise ValueError("Action head not initialized")
        
        # Convert observation to image
        image = self.capture_game_frame(observation)
        
        # Extract visual features
        visual_features = self.get_visual_features([image])
        
        # Predict action using trained head
        with torch.no_grad():
            action_logits = self.action_head(visual_features)
            action = torch.argmax(action_logits, dim=-1).item()
        
        return action, f"Head-only prediction: {action}"

    def extract_game_features(
        self, info: Dict
    ) -> Dict:  # Method to extract game features from info dict
        """
        Extract structured features from game state info based on ta.json schema

        Args:
            info: Game state info dictionary from environment

        Returns:
            Dictionary of structured game features
        """
        features = {  # Dictionary to store extracted game features
            # Player status (from ta.json)
            "agent_hp": info.get(
                "agent_hp", 176
            ),  # Get agent health points, default 176
            "agent_x": info.get("agent_x", 0),  # Get agent x-coordinate position
            "agent_y": info.get("agent_y", 0),  # Get agent y-coordinate position
            "agent_status": info.get("agent_status", 0),  # Get agent animation status
            "agent_victories": info.get("agent_victories", 0),  # Get agent wins count
            # Enemy status (from ta.json)
            "enemy_hp": info.get(
                "enemy_hp", 176
            ),  # Get enemy health points, default 176
            "enemy_x": info.get("enemy_x", 0),  # Get enemy x-coordinate position
            "enemy_y": info.get("enemy_y", 0),  # Get enemy y-coordinate position
            "enemy_status": info.get("enemy_status", 0),  # Get enemy animation status
            "enemy_victories": info.get("enemy_victories", 0),  # Get enemy wins count
            # Game status (from ta.json)
            "score": info.get("score", 0),  # Get current score
            "round_countdown": info.get(
                "round_countdown", 99
            ),  # Get time remaining in round
        }

        # Calculate derived features (only from available data)
        features["hp_advantage"] = (
            features["agent_hp"] - features["enemy_hp"]
        )  # Calculate health advantage
        features["distance"] = abs(
            features["agent_x"] - features["enemy_x"]
        )  # Calculate horizontal distance
        features["height_diff"] = (
            features["agent_y"] - features["enemy_y"]
        )  # Calculate vertical distance

        # Determine relative position
        if (
            features["agent_x"] < features["enemy_x"]
        ):  # If agent is to the left of enemy
            features["facing"] = "right"  # Agent should face right
        else:  # If agent is to the right of enemy
            features["facing"] = "left"  # Agent should face left

        return features  # Return the features dictionary

    def capture_game_frame(
        self, observation
    ) -> Image.Image:  # Method to convert observation to PIL Image
        """
        Convert game observation to PIL Image for vision model

        Args:
            observation: Game frame from environment (numpy array)

        Returns:
            PIL Image of the game frame
        """
        if isinstance(observation, np.ndarray):  # Check if observation is numpy array
            # Handle different observation formats
            if observation.shape == (1, 1, 1):  # Check for minimal observation shape
                # Single pixel observation - create a dummy RGB image
                dummy_frame = np.zeros(
                    (224, 320, 3), dtype=np.uint8
                )  # Create blank RGB frame
                return Image.fromarray(dummy_frame)  # Convert to PIL Image and return

            # Convert numpy array to PIL Image
            if observation.dtype != np.uint8:  # Check if values need scaling
                observation = (observation * 255).astype(
                    np.uint8
                )  # Scale to 0-255 and convert to uint8

            # Ensure proper shape for image
            if len(observation.shape) == 3 and observation.shape[2] in [
                3,
                4,
            ]:  # Check for RGB/RGBA format
                # RGB or RGBA image
                if observation.shape[2] == 4:  # Check if has alpha channel
                    observation = observation[
                        :, :, :3
                    ]  # Remove alpha channel, keep RGB only

                # pass obs to image obj's from array, to get image
                image = Image.fromarray(observation)  # Convert numpy array to PIL Image
            elif len(observation.shape) == 2:  # Check for grayscale format
                # Grayscale - convert to RGB
                image = Image.fromarray(observation).convert(
                    "RGB"
                )  # Convert grayscale to RGB
            else:  # Handle unexpected formats
                # Unexpected format - create dummy image
                dummy_frame = np.zeros(
                    (224, 320, 3), dtype=np.uint8
                )  # Create blank RGB frame
                return Image.fromarray(dummy_frame)  # Convert to PIL Image and return

            return image  # Return the processed PIL Image
        else:  # If not numpy array
            # If already PIL Image, return as-is
            return observation  # Return observation unchanged



    def get_action(
        self, observation, info: Dict, verbose: bool = False
    ) -> Tuple[int, str]:
        """
        Get action decision from Qwen2.5-VL with proper timing control
        Respects attack recovery frames and cooldowns

        Args:
            observation: Game frame (numpy array or PIL Image)
            info: Game state info from environment
            verbose: Whether to print reasoning

        Returns:
            Tuple of (action_number, reasoning_text)
        """
        self.frame_counter += 1
        self.frames_since_last_action += 1

        # the cool down reduced, so get ready for next action
        if self.action_cooldown > 0:
            self.action_cooldown -= 1

        # not in cool down, we can execute
        action_allowed = self.action_cooldown <= 0

        # 30 frames or 1 sec, we can do
        if (
            self.frame_counter % 30 == 0 or self.frame_counter == 60
        ) and action_allowed:
            # from obs to img
            image = self.capture_game_frame(observation)
            # so we have all the hp feature
            features = self.extract_game_features(info)

            # Update frame history for temporal context
            self.frame_history.append({"image": image, "features": features})
            if len(self.frame_history) > self.max_history_frames:
                self.frame_history.pop(0)  # Remove oldest frame

            # Use direct action prediction (no prompts)
            action, response = self.predict_action_direct(image)

            # Prevent repeating the same attack too many times (causes blocking)
            if action == self.last_action:
                self.action_repeat_count += 1
                if self.action_repeat_count > 3:  # If repeating more than 3 times
                    # Encourage variety by suggesting no action
                    old_action = action
                    action = 0  # NO_ACTION to break pattern
                    print(f"🔄 BREAKING REPEAT: {old_action} → {action} (NO_ACTION)")
                    self.action_repeat_count = 0
            else:
                self.action_repeat_count = 0

            # Set cooldown based on action recovery frames
            recovery_frames = self.action_frames.get(action, 10)
            self.action_cooldown = recovery_frames
            self.frames_since_last_action = 0
            self.last_executed_action = action

            # Cache the new decision
            self.last_action = action
            self.last_reasoning = response

        elif action_allowed:
            # If we're allowed to act but not on a thinking frame, use cached action
            action = self.last_action
            response = self.last_reasoning
        else:
            # We're in cooldown - must wait, use NO_ACTION
            action = 0  # NO_ACTION during recovery
            response = f"RECOVERY FRAMES: {self.action_cooldown} remaining from {self.action_meanings[self.last_executed_action]}"

        # Update action history
        action_name = self.action_meanings[action]
        self.action_history.append(action_name)

        # Keep history manageable
        if len(self.action_history) > 20:
            self.action_history = self.action_history[-20:]

        if verbose:
            model_status = (
                "NEW"
                if (self.frame_counter % 30 == 0 or self.frame_counter == 60)
                and action_allowed
                else "RECOVERY" if not action_allowed else "CACHED"
            )
            print(f"\n🚀 Qwen2.5-VL Timing Decision ({model_status}):")
            print(f"Frame: {self.frame_counter}")
            print(f"Action: {action} ({action_name})")
            if not action_allowed:
                print(
                    f"⏳ Cooldown: {self.action_cooldown} frames from {self.action_meanings[self.last_executed_action]}"
                )
            else:
                print(
                    f"Model Response: {response} {'(cached)' if model_status == 'CACHED' else ''}"
                )

        return action, response

    def reset(self):  # Method to reset agent state
        """Reset the agent state"""
        self.action_history = []  # Clear action history list
        self.last_features = {}  # Clear previous game features
        self.frame_counter = 0  # Reset frame counter to zero
        self.last_action = 0  # Reset last action to NO_ACTION
        self.last_reasoning = (
            "Initial state - match starting"  # Reset reasoning to initial state
        )
        self.attack_cycle_index = 0  # Reset attack cycle
        self.action_repeat_count = 0  # Reset action repeat counter
        self.last_distance = 0  # Reset last distance measurement
        self.action_cooldown = 0  # Reset action cooldown timer
        self.last_executed_action = 0  # Reset last executed action
        self.frames_since_last_action = 0  # Reset frame timing

    def train_head_simple(self, training_data, epochs=3, learning_rate=1e-4):
        """Simple head-only training method"""
        if not hasattr(self, 'action_head'):
            raise ValueError("Action head not initialized")
        
        import torch.nn as nn
        
        print(f"🚀 Training action head for {epochs} epochs...")
        
        # Setup training
        optimizer = torch.optim.AdamW(self.action_head.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        self.action_head.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for i, sample in enumerate(training_data):
                # Extract frame and action
                frame = np.array(sample['frame'], dtype=np.uint8)
                if len(frame.shape) == 1:
                    frame = frame.reshape((224, 320, 3))
                
                image = Image.fromarray(frame)
                action_target = torch.tensor([sample['action']], dtype=torch.long, device=self.device)
                
                # Get visual features
                visual_features = self.get_visual_features([image])
                
                # Forward pass
                optimizer.zero_grad()
                action_logits = self.action_head(visual_features)
                loss = criterion(action_logits, action_target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(action_logits, 1)
                correct += (predicted == action_target).sum().item()
                total += 1
                
                if (i + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}, Sample {i+1}, Loss: {loss.item():.4f}")
            
            accuracy = 100 * correct / total
            avg_loss = total_loss / len(training_data)
            print(f"✅ Epoch {epoch+1}/{epochs}: Loss {avg_loss:.4f}, Accuracy {accuracy:.2f}%")
        
        self.action_head.eval()
        print("🎯 Head-only training completed!")
        
        # Auto-save after training
        self.auto_save_head()

    def auto_load_head(self):
        """Auto-load action head from current directory if exists"""
        import os
        import torch
        
        if os.path.exists(self.action_head_path):
            try:
                print(f"📁 Auto-resuming from: {self.action_head_path}")
                self.action_head.load_state_dict(torch.load(self.action_head_path, map_location=self.device))
                print("✅ Action head loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load action head: {e}")
                print("🔄 Starting with fresh action head")
        else:
            print("🆕 No existing action head found - starting fresh")
    
    def auto_save_head(self):
        """Auto-save action head to current directory"""
        import torch
        
        try:
            torch.save(self.action_head.state_dict(), self.action_head_path)
            print(f"💾 Action head saved: {self.action_head_path}")
        except Exception as e:
            print(f"⚠️ Failed to save action head: {e}")

    def enable_online_learning(self, learning_rate=1e-5):
        """Enable online learning during gameplay"""
        import torch.nn as nn
        import itertools
        
        print("🔥 Enabling full model online learning...")
        self.online_learning = True
        
        # Setup optimizer for all trainable components
        param_groups = [self.action_head.parameters()]
        if hasattr(self, 'vision_projector'):
            param_groups.append(self.vision_projector.parameters())
        all_params = itertools.chain(*param_groups)
        self.optimizer = torch.optim.AdamW(all_params, lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"✅ Full model online learning enabled with lr={learning_rate}")
    
    def add_training_sample(self, observation, action, reward):
        """Add sample to training buffer for online learning"""
        if not self.online_learning:
            return
            
        # Convert observation to training format
        if hasattr(observation, 'flatten'):
            frame = observation.flatten().tolist()
        else:
            frame = observation
            
        sample = {
            'frame': frame,
            'action': int(action),
            'reward': float(reward)
        }
        
        self.training_buffer.append(sample)
        
        # Train when buffer is full
        if len(self.training_buffer) >= self.buffer_size:
            self._train_online_batch()
            self.training_buffer = []  # Clear buffer
    
    def _train_online_batch(self):
        """Train on current buffer of samples"""
        if not self.online_learning or len(self.training_buffer) == 0:
            return
            
        print(f"🚀 Full model online training on {len(self.training_buffer)} samples...")
        
        # Set models to training mode
        self.action_head.train()
        if hasattr(self, 'vision_projector'):
            self.vision_projector.train()
        total_loss = 0
        correct = 0
        
        for sample in self.training_buffer:
            # Prepare data
            frame = np.array(sample['frame'], dtype=np.uint8)
            if len(frame.shape) == 1:
                # Try to reshape to expected dimensions
                if len(frame) == 224 * 320 * 3:
                    frame = frame.reshape((224, 320, 3))
                else:
                    # Skip malformed frames
                    continue
                    
            image = Image.fromarray(frame)
            action_target = torch.tensor([sample['action']], dtype=torch.long, device=self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            visual_features = self.get_visual_features([image])
            action_logits = self.action_head(visual_features)
            loss = self.criterion(action_logits, action_target)
            
            # Debug info
            if total_loss == 0:  # First sample
                print(f"🔧 Debug - Action target: {action_target.item()}")
                print(f"🔧 Debug - Action logits range: {action_logits.min().item():.4f} to {action_logits.max().item():.4f}")
                print(f"🔧 Debug - Loss: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            if total_loss == 0:  # First sample
                grad_norm = sum(p.grad.norm().item() for p in self.action_head.parameters() if p.grad is not None)
                print(f"🔧 Debug - Gradient norm: {grad_norm:.6f}")
            
            self.optimizer.step()
            
            # Stats
            total_loss += loss.item()
            _, predicted = torch.max(action_logits, 1)
            correct += (predicted == action_target).sum().item()
            
            # Debug predictions
            if total_loss <= loss.item():  # First sample
                print(f"🔧 Debug - Predicted: {predicted.item()}, Target: {action_target.item()}")
        
        accuracy = 100 * correct / len(self.training_buffer)
        avg_loss = total_loss / len(self.training_buffer)
        print(f"📊 Full model batch: Loss {avg_loss:.4f}, Accuracy {accuracy:.1f}%")
        
        # Set models back to eval mode
        self.action_head.eval()
        if hasattr(self, 'vision_projector'):
            self.vision_projector.eval()
        
        # Auto-save full model after each online training batch
        self.save_model()


# Demo functions removed - use play.py for gameplay


# qwen_agent.py is now an isolated agent class only
# Use play.py for gameplay and inference
if __name__ == "__main__":
    import argparse
    import json
    import os
    
    parser = argparse.ArgumentParser(description="Qwen Street Fighter 2 Agent Training")
    parser.add_argument("--train", action="store_true", help="Start training")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (copy model from cache)")
    parser.add_argument("--collect-data", action="store_true", help="Collect training data by playing")
    parser.add_argument("--data-path", type=str, default="./data/sf2_training_data.json", help="Training data path")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes for data collection")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="./sf2_action_head.pth", help="Model save path")
    
    args = parser.parse_args()
    
    if args.collect_data:
        print("🎮 Starting data collection gameplay...")
        
        # Import gameplay modules
        import retro
        from discretizer import StreetFighter2Discretizer
        
        # Create environment
        game = retro.make(
            "StreetFighterIISpecialChampionEdition-Genesis",
            state="ken_bison_12.state", 
            use_restricted_actions=retro.Actions.FILTERED,
        )
        env = StreetFighter2Discretizer(game)
        
        # Create agent for demonstration
        agent = QwenStreetFighterAgent()
        
        # Data collection
        training_data = []
        
        for episode in range(args.episodes):
            print(f"\n🏁 Episode {episode + 1}/{args.episodes} - Data Collection")
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            agent.reset()
            step = 0
            max_steps = 1000
            
            while step < max_steps:
                # Render game UI
                env.render()
                
                # Extract real game info from environment
                try:
                    # Try to get real game state from retro environment
                    game_state = env.unwrapped.data
                    info = {
                        'agent_hp': game_state.get('health', 150),
                        'enemy_hp': game_state.get('enemy_health', 120),
                        'agent_x': game_state.get('x', 100),
                        'agent_y': game_state.get('y', 200),
                        'enemy_x': game_state.get('enemy_x', 200),
                        'enemy_y': game_state.get('enemy_y', 200),
                        'score': game_state.get('score', 0),
                        'round_countdown': game_state.get('timer', 99)
                    }
                except (AttributeError, KeyError):
                    # Fallback if real data unavailable
                    info = {
                        'agent_hp': 150,
                        'enemy_hp': 120, 
                        'agent_x': 100,
                        'agent_y': 200,
                        'enemy_x': 200,
                        'enemy_y': 200,
                        'score': 0,
                        'round_countdown': 99
                    }
                
                # Get action from agent
                action, reasoning = agent.get_action(obs, info, verbose=False)
                
                # Store training sample every 10 frames
                if step % 10 == 0:
                    frame_data = {
                        'frame': obs.flatten().tolist() if hasattr(obs, 'flatten') else obs,
                        'action': int(action),
                        'step': step
                    }
                    training_data.append(frame_data)
                
                # Take step
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, done, truncated, _ = result
                else:
                    obs, reward, done, truncated = result
                
                if done or step >= max_steps:
                    break
                    
                step += 1
            
            print(f"✅ Episode {episode + 1} completed: {step} steps, {len(training_data)} samples collected")
        
        env.close()
        
        # Save training data
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        with open(args.data_path, 'w') as f:
            json.dump(training_data, f)
        print(f"💾 Training data saved: {len(training_data)} samples -> {args.data_path}")
        
    elif args.train:
        print("🚀 Starting Qwen agent training...")
        
        # Create agent
        agent = QwenStreetFighterAgent()
        
        # Resume from checkpoint if specified
        if args.resume:
            if os.path.exists(args.resume):
                print(f"📁 Resuming from: {args.resume}")
                import torch
                agent.action_head.load_state_dict(torch.load(args.resume))
            else:
                print(f"❌ Checkpoint not found: {args.resume}")
                exit(1)
        
        # Load training data
        if os.path.exists(args.data_path):
            with open(args.data_path, 'r') as f:
                training_data = json.load(f)
            print(f"📊 Loaded {len(training_data)} training samples")
        else:
            print(f"❌ Training data not found: {args.data_path}")
            exit(1)
        
        # Train
        agent.train_head_simple(training_data, epochs=args.epochs, learning_rate=args.learning_rate)
        
        # Save model
        import torch
        torch.save(agent.action_head.state_dict(), args.save_path)
        print(f"💾 Model saved to: {args.save_path}")
        
    else:
        print("🔥 Starting online learning gameplay...")
        
        # Import gameplay modules
        import retro
        from discretizer import StreetFighter2Discretizer
        
        # Create environment
        game = retro.make(
            "StreetFighterIISpecialChampionEdition-Genesis",
            state="ken_bison_12.state", 
            use_restricted_actions=retro.Actions.FILTERED,
        )
        env = StreetFighter2Discretizer(game)
        
        # Create agent with fresh start option
        agent = QwenStreetFighterAgent(fresh_start=args.fresh)
        
        for episode in range(args.episodes):
            print(f"\n🏁 Episode {episode + 1}/{args.episodes} - Online Learning")
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            agent.reset()
            step = 0
            max_steps = 1000
            total_reward = 0
            
            while step < max_steps:
                # Render game UI
                env.render()
                
                # Extract real game info from environment
                try:
                    # Try to get real game state from retro environment
                    game_state = env.unwrapped.data
                    info = {
                        'agent_hp': game_state.get('health', 150),
                        'enemy_hp': game_state.get('enemy_health', 120),
                        'agent_x': game_state.get('x', 100),
                        'agent_y': game_state.get('y', 200),
                        'enemy_x': game_state.get('enemy_x', 200),
                        'enemy_y': game_state.get('enemy_y', 200),
                        'score': game_state.get('score', 0),
                        'round_countdown': game_state.get('timer', 99)
                    }
                except (AttributeError, KeyError):
                    # Fallback if real data unavailable
                    info = {
                        'agent_hp': 150,
                        'enemy_hp': 120, 
                        'agent_x': 100,
                        'agent_y': 200,
                        'enemy_x': 200,
                        'enemy_y': 200,
                        'score': 0,
                        'round_countdown': 99
                    }
                
                # Get action from agent
                action, reasoning = agent.get_action(obs, info, verbose=False)
                
                # Take step
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, done, truncated, _ = result
                else:
                    obs, reward, done, truncated = result
                
                total_reward += reward
                
                # Add sample for online learning (always active)
                agent.add_training_sample(obs, action, reward)
                
                if done or step >= max_steps:
                    break
                    
                step += 1
            
            print(f"✅ Episode {episode + 1} completed: {step} steps, reward: {total_reward:.2f}")
        
        env.close()
        
        # Save the trained model
        agent.save_model()
