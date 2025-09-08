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
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info

# Import NumPy for numerical array operations
import numpy as np

# for img
from PIL import Image
import re  # Import regular expressions for text pattern matching
from typing import Dict, Tuple  # Import typing hints for better code documentation
import time  # Import time module for sleep delays
import argparse  # Import argument parser module
import math
from collections import defaultdict


class QwenStreetFighterAgent:  # Define main agent class for Street Fighter 2 AI
    """
    Qwen-powered agent for Street Fighter 2
    Uses existing wrapper.py environment without modifications
    """

    # agent init
    def __init__(
        self,
        model_path: str = "/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-7B-Instruct-AWQ",
    ):  # Constructor method for agent initialization
        """
        Initialize the Qwen agent

        Args:
            model_path: Path to Qwen model (local or HuggingFace)
        """
        # Initialize Qwen model
        print(f"🤖 Loading Qwen AWQ model from: {model_path}")  # Print model loading status

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
            "📁 Step 2/2: Loading Qwen2.5-VL model from cache..."
        )  # Print loading status for model

        # Load AWQ quantized model using AutoAWQ
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.float16,  # AWQ uses fp16 for activations
            local_files_only=True,
        )
        print(
            f"✅ Qwen AWQ model loaded successfully on {self.device}"
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

    def create_unified_prompt(
        self, features: Dict
    ) -> str:  # Method to create single comprehensive prompt
        """
        Create concise unified prompt for fighting game analysis and action selection

        Args:
            features: Game state features from ta.json

        Returns:
            Concise prompt for game state analysis and action decision
        """
        # Calculate key metrics
        distance = features["distance"]
        hp_diff = features["agent_hp"] - features["enemy_hp"]
        
        # Determine tactical situation
        range_context = "CLOSE" if distance < 60 else "MID" if distance < 120 else "FAR"
        hp_status = "WINNING" if hp_diff > 20 else "LOSING" if hp_diff < -20 else "EVEN"
        
        # Movement context from frame history
        movement_context = ""
        if len(self.frame_history) >= 2:
            prev_distance = self.frame_history[-2]["features"]["distance"]
            distance_change = distance - prev_distance
            if distance_change < -10:
                movement_context = "ENEMY RUSHING"
            elif distance_change > 10:
                movement_context = "ENEMY RETREATING"
            else:
                movement_context = "NEUTRAL"

        # Determine facing direction based on enemy position
        facing_right = features["agent_x"] < features["enemy_x"]
        facing_dir = "RIGHT" if facing_right else "LEFT"
        
        # Special moves based on facing direction
        hadoken = "38=HADOKEN_RIGHT" if facing_right else "41=HADOKEN_LEFT"
        dragon_punch = "39=DRAGON_PUNCH_RIGHT" if facing_right else "42=DRAGON_PUNCH_LEFT"  
        hurricane_kick = "40=HURRICANE_KICK_RIGHT" if facing_right else "43=HURRICANE_KICK_LEFT"

        prompt = f"""Street Fighter 2 - Frame {self.frame_counter}

SITUATION:
My HP: {features['agent_hp']} | Enemy HP: {features['enemy_hp']} | Status: {hp_status}
Range: {distance}px ({range_context}) | Movement: {movement_context} | Facing: {facing_dir}

FIGHTING PRINCIPLES:
- If enemy rushing → block/counter (actions 2,3 or defensive moves)
- If enemy vulnerable → attack (punches 9-20, kicks 21-37)
- At distance → fireball ({hadoken.split('=')[0]})
- Anti-air/wakeup → uppercut ({dragon_punch.split('=')[0]})  
- Close pressure → spinning kick ({hurricane_kick.split('=')[0]})
- Mix up attacks to avoid being predictable

ACTIONS: 0=none, 1=up, 2=down, 3=left, 6=right, 9-20=punches, 21-37=kicks
SPECIALS: {hadoken}, {dragon_punch}, {hurricane_kick}

Analyze the 8 frames and choose your next move. Output only the action number (0-43):"""

        return prompt

    def query_qwen_vl(
        self, images: list, prompt: str
    ) -> str:  # Method to query Qwen2.5-VL model with frame stack
        """
        Query Qwen2.5-VL model with frame stack and prompt

        Args:
            images: List of game frames as PIL Images (frame stack)
            prompt: Text prompt for analysis

        Returns:
            Model's response containing reasoning and action
        """
        # Create content list with multiple images (frame stack) + text
        content = []

        # content has image obj x N + text prompt
        for i, img in enumerate(images):
            content.append({"type": "image", "image": img})

        # Add text prompt at the end
        content.append({"type": "text", "text": prompt})

        # Create messages for Qwen2.5-VL format
        messages = [  # Create message list in chat format
            {  # User message containing multiple images and text
                "role": "user",  # Set role as user
                "content": content,  # Content list with frame stack + text
            }
        ]

        # merge images (frame stack) and prompt and pass to vl model
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Debug the inputs being generated
        inputs = self.processor(text=text_input, images=images, return_tensors="pt")

        # DEBUG: Print what we're actually sending to the model
        print(f"\n🔧 DEBUG INPUT PROCESSING:")
        print(f"Text input length: {len(text_input)}")
        print(f"Frame stack size: {len(images)} images")
        if len(images) > 0:
            print(f"Image type: {type(images[0])}")
            print(f"Image size: {images[0].size}")
        print(f"Input keys: {list(inputs.keys())}")
        if "pixel_values" in inputs:
            print(
                f"✅ pixel_values shape: {inputs['pixel_values'].shape} (includes {len(images)} frames)"
            )
        else:
            print(f"❌ NO pixel_values in inputs!")
        print(f"Text preview: {text_input[:200]}...")

        # Move tensors to device and filter out unsupported parameters
        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
            if k != "pixel_attention_mask"
        }  # Filter out pixel_attention_mask

        # Generate response with maximum speed optimization
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model.generate(  # Generate response from model
                **inputs,  # Pass all input tensors
                max_new_tokens=50,  # Moderate response length for action selection
                do_sample=False,  # Use greedy decoding (deterministic)
                pad_token_id=self.processor.tokenizer.pad_token_id,  # Set padding token
                num_beams=1,  # Single beam for maximum speed
                use_cache=True,  # Enable KV cache for speed
            )

        # Decode response
        full_response = self.processor.decode(
            outputs[0], skip_special_tokens=True
        )  # Decode tokens to text

        # Extract only the assistant's response (after the last "assistant" marker)
        if "assistant" in full_response:
            response = full_response.split("assistant")[-1].strip()
        else:
            response = full_response.strip()

        print(f"\n🤖 FULL MODEL RESPONSE:\n{full_response}")
        print(f"\n📝 EXTRACTED RESPONSE: '{response}'")
        return response

    def parse_action_from_response(self, response: str) -> int:
        """
        Parse the action number from Qwen's response

        Args:
            response: Raw response from Qwen

        Returns:
            Action number (0-43), defaults to intelligent fallback if parsing fails
        """
        try:
            # Clean the response - remove extra whitespace and newlines
            response = response.strip()

            # Look for standalone numbers first (most direct answer)
            standalone_number = re.search(r"^\s*(\d+)\s*$", response, re.MULTILINE)
            if standalone_number:
                action = int(standalone_number.group(1))
                if 0 <= action < self.num_actions:
                    print(f"✅ FOUND STANDALONE ACTION: {action}")
                    return action

            # Look for "Action: X" or "Action X" pattern
            action_match = re.search(
                r"(?:Action:?\s*|^)(\d+)", response, re.IGNORECASE | re.MULTILINE
            )
            if action_match:
                action = int(action_match.group(1))
                if 0 <= action < self.num_actions:
                    print(f"✅ FOUND ACTION PATTERN: {action}")
                    return action

            # Look for numbers that could be actions - ALL ACTIONS ALLOWED
            numbers = re.findall(r"\b(\d+)\b", response)
            
            # Check for any valid action number
            for num_str in numbers:
                num = int(num_str)
                # Skip numbers likely from "Street Fighter 2" or other context
                if num == 2 and "Fighter 2" in response:
                    continue
                if 0 <= num < self.num_actions:
                    print(f"✅ FOUND VALID ACTION: {num}")
                    return num

            # If no numbers found, try to infer action from keywords
            response_lower = response.lower()

            # Map keywords to actions
            if any(word in response_lower for word in ["punch", "hit", "attack", "strike"]):
                print(f"🔍 INFERRED FROM 'punch': 9 (LIGHT_PUNCH)")
                return 9  # LIGHT_PUNCH
            elif any(word in response_lower for word in ["kick"]):
                print(f"🔍 INFERRED FROM 'kick': 21 (LIGHT_KICK)")
                return 21  # LIGHT_KICK
            elif any(word in response_lower for word in ["jump", "up"]):
                print(f"🔍 INFERRED FROM 'jump': 1 (UP)")
                return 1  # UP
            elif any(word in response_lower for word in ["right", "forward", "advance"]):
                print(f"🔍 INFERRED FROM 'right': 6 (RIGHT)")
                return 6  # RIGHT
            elif any(word in response_lower for word in ["left", "back", "retreat", "block"]):
                print(f"🔍 INFERRED FROM 'left/block': 3 (LEFT)")
                return 3  # LEFT
            elif any(word in response_lower for word in ["crouch", "duck", "down", "low"]):
                print(f"🔍 INFERRED FROM 'crouch': 2 (DOWN)")
                return 2  # DOWN
            elif any(word in response_lower for word in ["hadoken", "fireball"]):
                print(f"🔍 INFERRED FROM 'hadoken': 38 (HADOKEN_RIGHT)")
                return 38  # HADOKEN_RIGHT
            elif any(word in response_lower for word in ["uppercut", "dragon"]):
                print(f"🔍 INFERRED FROM 'uppercut': 39 (DRAGON_PUNCH_RIGHT)")
                return 39  # DRAGON_PUNCH_RIGHT

            # Simple fallback - default to no action
            print(f"⚠️ NO KEYWORDS FOUND - DEFAULTING TO: 0 (NO_ACTION)")
            return 0

        except Exception as e:
            print(f"❌ Action parsing failed: {e}")
            return 0  # Default to no action

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

            # Create unified prompt with all game state info and actions
            prompt = self.create_unified_prompt(features)

            # Prepare frame stack for vision model
            frame_stack = []
            if len(self.frame_history) >= 8:
                # Use last 8 frames as stack
                for frame_data in self.frame_history[-8:]:
                    frame_stack.append(frame_data["image"])
            else:
                # If we don't have 8 frames yet, pad with current frame
                for frame_data in self.frame_history:
                    frame_stack.append(frame_data["image"])
                # Pad with current frame to reach 8 frames
                while len(frame_stack) < 8:
                    frame_stack.append(image)

            # Use frame stack for temporal understanding
            response = self.query_qwen_vl(frame_stack, prompt)

            # Parse action number from model response
            action = self.parse_action_from_response(response)

            # Prevent repeating the same attack too many times (causes blocking)
            if action == self.last_action:
                self.action_repeat_count += 1
                if self.action_repeat_count > 3:  # If repeating more than 3 times
                    # Encourage variety by suggesting no action
                    old_action = action
                    action = 0  # NO_ACTION to break pattern
                    print(
                        f"🔄 BREAKING REPEAT: {old_action} → {action} (NO_ACTION)"
                    )
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


# Demo functions from demo_qwen.py
def demo_qwen_gameplay(
    model_path: str = "/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-7B-Instruct-AWQ",  # Function to demo Qwen agent gameplay
    episodes: int = 3,  # Default number of episodes to play
    render: bool = True,  # Default to render game visuals
    verbose: bool = True,
):  # Default to verbose output
    """
    Demo Qwen agent playing Street Fighter 2

    Args:
        model_path: Path to Qwen model
        episodes: Number of episodes to play
        render: Whether to render the game
        verbose: Whether to print detailed reasoning
    """
    from wrapper import make_env  # Import environment creation function from wrapper

    print("🥊 Starting Qwen Street Fighter Demo")  # Print demo start message
    print(f"Model: {model_path}")  # Print model path being used
    print(f"Episodes: {episodes}")  # Print number of episodes to play
    print("-" * 50)  # Print separator line

    # Create environment and agent
    env = make_env()  # Create Street Fighter 2 environment
    agent = QwenStreetFighterAgent(model_path)  # Create Qwen agent with specified model

    wins = 0  # Initialize win counter
    total_rewards = []  # Initialize list to store episode rewards

    try:  # Try to run episodes with error handling
        for episode in range(episodes):  # Loop through each episode
            print(f"\n🎮 Episode {episode + 1}/{episodes}")  # Print episode header
            print("=" * 30)  # Print episode separator

            # Reset environment and agent
            obs, info = env.reset()  # Reset environment and get initial observation
            agent.reset()  # Reset agent internal state

            episode_reward = 0  # Initialize episode reward accumulator
            steps = 0  # Initialize step counter
            max_steps = 5000  # Allow full match completion with step limit

            while steps < max_steps:  # Main episode loop
                if render:  # Check if should render visuals
                    env.render()  # Render game screen
                    time.sleep(0.05)  # Slow down for visibility (50ms delay)

                # Get action from Qwen vision agent
                action, reasoning = agent.get_action(
                    obs, info, verbose=verbose
                )  # Get AI action decision

                # Take step in environment
                obs, reward, done, truncated, info = env.step(
                    action
                )  # Execute action and get results
                episode_reward += reward  # Add reward to episode total
                steps += 1  # Increment step counter

                # Check if episode ended
                if done or truncated:  # Check if episode is finished
                    agent_won = info.get(
                        "agent_won", False
                    )  # Check if agent won the match
                    if agent_won:  # If agent won
                        wins += 1  # Increment win counter
                        print(f"🏆 Won episode {episode + 1}!")  # Print victory message
                    else:  # If agent lost
                        print(f"💀 Lost episode {episode + 1}")  # Print loss message

                    print(
                        f"Steps: {steps}, Reward: {episode_reward:.2f}"
                    )  # Print episode statistics
                    break  # Exit episode loop

            total_rewards.append(episode_reward)  # Add episode reward to total list

            if not render:  # If not rendering visuals
                print(
                    f"Episode {episode + 1} completed - Reward: {episode_reward:.2f}"
                )  # Print completion message

    except KeyboardInterrupt:  # Handle user interruption
        print("\n⚠️  Demo interrupted by user")  # Print interruption message

    finally:  # Always execute cleanup
        env.close()  # Close environment properly

    # Summary
    print("\n" + "=" * 50)  # Print summary header separator
    print("🏁 DEMO SUMMARY")  # Print summary title
    print("=" * 50)  # Print summary separator
    print(f"Episodes Played: {len(total_rewards)}")  # Print total episodes played
    print(f"Wins: {wins}")  # Print total wins
    print(
        f"Win Rate: {wins/len(total_rewards)*100:.1f}%" if total_rewards else "N/A"
    )  # Calculate and print win rate
    print(
        f"Average Reward: {sum(total_rewards)/len(total_rewards):.2f}"
        if total_rewards
        else "N/A"
    )  # Calculate and print average reward


def test_qwen_simple():  # Function for simple agent testing
    """Simple test of Qwen agent without full gameplay"""
    from wrapper import make_env  # Import environment creation function from wrapper

    print("🧪 Simple Qwen Agent Test")  # Print test header
    print("-" * 30)  # Print separator line

    # Create environment
    env = make_env()  # Create Street Fighter 2 environment
    obs, info = env.reset()  # Reset environment and get initial state

    # Create agent
    agent = QwenStreetFighterAgent(
        "/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-7B-Instruct-AWQ"
    )  # Create agent with 7B AWQ model

    # Test a few decisions
    for i in range(3):  # Loop through 3 test iterations
        print(f"\nTest {i+1}:")  # Print test iteration number
        action, reasoning = agent.get_action(
            obs, info, verbose=True
        )  # Get action from agent

        # so the env -> obs -> frame
        obs, reward, done, truncated, info = env.step(
            action
        )  # Execute action in environment
        print(f"Reward: {reward:.3f}")  # Print reward received

        if done:  # Check if episode ended
            break  # Exit test loop if done

    env.close()  # Close environment
    print("\n✅ Simple test completed")  # Print test completion message


# Test script
if __name__ == "__main__":  # Check if script is run directly
    parser = argparse.ArgumentParser(
        description="Qwen Street Fighter Demo (7B AWQ)"
    )  # Create argument parser
    parser.add_argument(
        "--model",
        type=str,
        default="/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-7B-Instruct-AWQ",  # 7B AWQ model path argument
        help="Qwen2.5-VL AWQ model local path",
    )  # Help text for model argument
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,  # Episodes count argument
        help="Number of episodes to play",
    )  # Help text for episodes argument
    parser.add_argument(
        "--no-render",
        action="store_true",  # No render flag argument
        help="Disable rendering for faster execution",
    )  # Help text for no-render argument
    parser.add_argument(
        "--quiet",
        action="store_true",  # Quiet mode flag argument
        help="Disable verbose reasoning output",
    )  # Help text for quiet argument
    parser.add_argument(
        "--test-only",
        action="store_true",  # Test only flag argument
        help="Run simple test instead of full demo",
    )  # Help text for test-only argument

    args = parser.parse_args()  # Parse command line arguments

    if args.test_only:  # Check if test-only mode requested
        test_qwen_simple()  # Run simple test function
    else:  # Otherwise run full demo
        demo_qwen_gameplay(  # Call main demo function
            model_path=args.model,  # Pass model path from arguments
            episodes=args.episodes,  # Pass episode count from arguments
            render=not args.no_render,  # Invert no-render flag for render parameter
            verbose=not args.quiet,  # Invert quiet flag for verbose parameter
        )
