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
        model_path: str = "/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-3B-Instruct",
    ):  # Constructor method for agent initialization
        """
        Initialize the Qwen agent

        Args:
            model_path: Path to Qwen model (local or HuggingFace)
        """
        # Initialize Qwen model
        print(f"🤖 Loading Qwen model from: {model_path}")  # Print model loading status

        # device cuda
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Set device to GPU if available, else CPU

        # Multi-frame context for temporal understanding - 4 frame stack
        self.frame_history = []  # Store recent frames for temporal context
        self.max_history_frames = 4  # Keep last 4 frames for context (frame stacking)

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

        # Load model without quantization for debugging vision capability
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.float16,  # Use same dtype as working example
            local_files_only=True,
        )
        print(
            f"✅ Qwen model loaded successfully on {self.device}"
        )  # Print successful loading message

        # Action space - matches discretizer.py (keep all actions but focus AI on basic ones)
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

        # Action variety system to prevent blocking - SIMPLIFIED BASIC ACTIONS ONLY
        self.attack_cycle_index = 0  # Cycle through different attacks
        self.aggressive_attacks = [
            9,
            21,
            3,
            6,
            1,
            2,
        ]  # Basic: punch, kick, left, right, jump, crouch
        self.action_repeat_count = 0  # Count how many times same action repeated
        self.last_distance = 0  # Store previous distance between characters

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
        Create single unified prompt for both game analysis and action selection

        Args:
            features: Game state features from ta.json

        Returns:
            Comprehensive prompt for game state analysis and action decision
        """
        # Calculate position and health differences
        x_diff = (
            features["agent_x"] - features["enemy_x"]
        )  # Positive = agent is right of enemy
        y_diff = (
            features["agent_y"] - features["enemy_y"]
        )  # Positive = agent is above enemy
        hp_diff = (
            features["agent_hp"] - features["enemy_hp"]
        )  # Positive = agent has more health

        # Determine positioning and health context
        distance_text = "close" if features["distance"] < 60 else "far"
        position_context = ""
        if abs(x_diff) > 20:  # Significant horizontal difference
            position_context = "left" if x_diff < 0 else "right"

        health_context = ""
        if abs(hp_diff) > 30:  # Significant health difference
            health_context = "winning" if hp_diff > 0 else "losing"

        # Get enemy status early for pattern detection
        enemy_status = features.get("enemy_status", 0)
        agent_status_val = features.get("agent_status", 0)

        # Add temporal context from 4-frame history stack
        history_context = ""
        if len(self.frame_history) >= 2:
            # Compare with previous frame
            prev_features = self.frame_history[-2]["features"]
            hp_change = features["agent_hp"] - prev_features["agent_hp"]
            enemy_hp_change = features["enemy_hp"] - prev_features["enemy_hp"]
            distance_change = features["distance"] - prev_features["distance"]

            # has 4 frames vs has less than 4 frames
            if len(self.frame_history) >= 4:
                oldest_features = self.frame_history[0]["features"]
                total_hp_change = features["agent_hp"] - oldest_features["agent_hp"]
                total_enemy_hp_change = (
                    features["enemy_hp"] - oldest_features["enemy_hp"]
                )

                history_context = f"""
🎬 4-FRAME SEQUENCE ANALYSIS:
Frame-to-frame: Agent HP {hp_change:+d}, Enemy HP {enemy_hp_change:+d}
Movement: {distance_change:+d}px ({'🚨 ENEMY APPROACHING!' if distance_change < -10 else '🏃 Enemy retreating' if distance_change > 10 else '⚖️ Neutral movement'})
4-Frame Trend: My HP {total_hp_change:+d}, Enemy HP {total_enemy_hp_change:+d}
Battle Momentum: {'⚠️ TAKING DAMAGE - DEFEND!' if total_hp_change < -5 else '💪 DEALING DAMAGE - KEEP PRESSURE!' if total_enemy_hp_change < -5 else '⚔️ NEUTRAL FIGHT'}

🔮 PATTERN DETECTION:
• Distance changing by {distance_change:+d}px suggests: {'🚨 INCOMING ATTACK!' if distance_change < -15 else '📉 ENEMY RETREATING' if distance_change > 15 else '🎯 POSITIONING'}
• Enemy status sequence suggests: {'⚡ POSSIBLE SPECIAL MOVE STARTUP' if enemy_status != oldest_features.get('enemy_status', 0) else '👤 NORMAL MOVEMENT'}"""
            else:
                history_context = f"""
🎬 BUILDING 4-FRAME STACK ({len(self.frame_history)}/4 frames):
Recent Change: Agent HP {hp_change:+d}, Enemy HP {enemy_hp_change:+d}
Movement: {distance_change:+d}px ({'🚨 APPROACHING' if distance_change < -5 else '🏃 RETREATING' if distance_change > 5 else '⚖️ NEUTRAL'})
Early Pattern: {'💥 TAKING HITS' if hp_change < 0 else '🎯 LANDING HITS' if enemy_hp_change < 0 else '⚔️ NEUTRAL EXCHANGE'}"""

        # Determine health status clearly
        agent_status = (
            "CRITICAL"
            if features["agent_hp"] < 50
            else "LOW" if features["agent_hp"] < 100 else "HEALTHY"
        )
        enemy_status = (
            "CRITICAL"
            if features["enemy_hp"] < 50
            else "LOW" if features["enemy_hp"] < 100 else "HEALTHY"
        )
        tactical_status = (
            "WINNING" if hp_diff > 30 else "LOSING" if hp_diff < -30 else "EVEN"
        )

        # Always prioritize aggressive approach - this is a fighting game!
        my_approach = "ULTRA_AGGRESSIVE"  # Always be aggressive
        enemy_threat = (
            "TARGET"
            if enemy_status == "HEALTHY"
            else "WEAK_TARGET" if enemy_status == "LOW" else "FINISH_HIM"
        )

        # Basic combat strategy based on distance
        distance_strategy = ""
        if features["distance"] < 40:  # Close range
            distance_strategy = (
                "CLOSE RANGE: PUNCH (9), KICK (21), or BLOCK (3=away, 6=toward)!"
            )
        elif features["distance"] < 80:  # Medium range
            distance_strategy = (
                "MEDIUM RANGE: MOVE CLOSER (6=RIGHT) or JUMP (1) to close gap!"
            )
        else:  # Long range
            distance_strategy = (
                "LONG RANGE: MOVE FORWARD (6=RIGHT) or JUMP (1) to get closer!"
            )

        # 4-FRAME PATTERN ANALYSIS - Predict enemy moves and counter!

        action_recommendations = f"""
🔮 4-FRAME SEQUENCE ANALYSIS - PREDICT THE FUTURE! 🔮

🎯 LOOK AT ALL 4 FRAMES TOGETHER - What pattern do you see?

📊 CURRENT SITUATION:
- MY HP: {features['agent_hp']} | ENEMY HP: {features['enemy_hp']}
- Distance: {features['distance']}px | Position: {'🔥 Close' if features['distance'] < 60 else '⚖️ Medium' if features['distance'] < 120 else '🏃 Far'}
- MY Status: {agent_status_val} | ENEMY Status: {enemy_status}
- HP Advantage: {hp_diff:+d}

🔍 VISUAL PATTERN ANALYSIS (analyze what you SEE in the 4 frames):
Look for these visual cues across the frame sequence:
• Position changes: Enemy moving toward/away from you?
• Height changes: Enemy crouching then rising (attack startup)?
• Rapid movement: Sudden position shifts (special moves)?
• Animation changes: Different poses between frames?
• Distance trends: Getting closer/farther over time?

🎯 ADAPTIVE COUNTER-STRATEGY:
• If you observe rapid approach → Use 3=LEFT (block) or 21=KICK (space)
• If you see upward movement → Use 2=CROUCH (avoid) or 9=PUNCH (anti-air)
• If you notice retreat pattern → Use 6=RIGHT (chase) or 1=JUMP (close)
• If distance stable + enemy animation change → Prepare counter-attack
• Trust your visual analysis - adapt based on what YOU see in the frames

⚡ BASIC COUNTER-ATTACKS:
- 9=PUNCH: Quick attack when enemy vulnerable
- 21=KICK: Keep distance, interrupt enemy moves
- 3=LEFT: Block/retreat from dangerous attacks  
- 6=RIGHT: Advance when enemy is recovering
- 1=JUMP: Avoid low attacks or close distance
- 2=CROUCH: Avoid high attacks, prepare counter
"""

        prompt = f"""🎬 4-FRAME SEQUENCE ANALYSIS - STREET FIGHTER 2 PREDICTION:

🔍 ANALYZE ALL 4 FRAMES TOGETHER:
Current Distance: {features['distance']}px ({distance_text})
Position: Agent {abs(x_diff)}px {"L" if x_diff < 0 else "R"} of enemy
MY Health: {features['agent_hp']} ({agent_status})
MY Status: {features['agent_status']} (watch my animation state)
ENEMY Health: {features['enemy_hp']} ({enemy_status})  
ENEMY Status: {features['enemy_status']} (CRITICAL - watch for attack patterns!)
Battle Status: {tactical_status} (My HP - Enemy HP = {hp_diff:+d}){history_context}

{distance_strategy}

🔮 PATTERN RECOGNITION MISSION:{action_recommendations}

🎯 BASIC ACTION ARSENAL:
MOVEMENT: 3=LEFT 6=RIGHT 1=JUMP 2=CROUCH
ATTACKS: 9=PUNCH 21=KICK
DEFENSE: 3=LEFT (block away) 2=CROUCH (low block)

⚡ PREDICTION-BASED STRATEGY:
- Watch 4-frame sequence for enemy attack startup
- Counter predicted moves with appropriate basic action
- Use movement to avoid, attacks to punish, blocks to defend

CRITICAL INSTRUCTION: ANALYZE THE 4 FRAMES FOR PATTERNS, THEN RESPOND WITH ONE NUMBER!
USE ONLY BASIC ACTIONS: 0, 1, 2, 3, 6, 9, 21
DO NOT EXPLAIN. DO NOT USE WORDS. JUST OUTPUT THE PREDICTED COUNTER-ACTION NUMBER.

🔮 PREDICTION EXAMPLES:
- See enemy rushing forward → 3=LEFT_BLOCK or 2=CROUCH
- See enemy jumping → 2=CROUCH or 9=PUNCH (anti-air)
- See enemy retreating → 6=RIGHT (follow) or 1=JUMP (close gap)
- See enemy vulnerable → 9=PUNCH or 21=KICK (attack!)

YOUR PREDICTIVE COUNTER-ACTION (0,1,2,3,6,9,21): """

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

            # Look for numbers that could be actions - PRIORITIZE BASIC ACTIONS
            numbers = re.findall(r"\b(\d+)\b", response)
            basic_actions = [
                0,
                1,
                2,
                3,
                6,
                9,
                21,
            ]  # NO_ACTION, UP, DOWN, LEFT, RIGHT, PUNCH, KICK

            # First check for basic actions
            for num_str in numbers:
                num = int(num_str)
                # Skip numbers likely from "Street Fighter 2" or other context
                if num == 2 and "Fighter 2" in response:
                    continue
                if num in basic_actions:
                    print(f"✅ FOUND BASIC ACTION: {num}")
                    return num

            # If no basic actions found, fall back to any valid action
            for num_str in numbers:
                num = int(num_str)
                if num == 2 and "Fighter 2" in response:
                    continue
                if 0 <= num < self.num_actions:
                    print(
                        f"⚠️ FOUND NON-BASIC ACTION: {num} - converting to basic equivalent"
                    )
                    # Convert complex actions to basic equivalents
                    if num in [17, 13, 9]:  # Any punch -> basic punch
                        return 9
                    elif num in [32, 26, 21]:  # Any kick -> basic kick
                        return 21
                    elif num in [4, 7]:  # Diagonal jumps -> basic jump
                        return 1
                    elif num in [5, 8]:  # Crouch movements -> basic crouch
                        return 2
                    else:
                        return num  # Use as-is if no basic equivalent

            # If no numbers found, try to infer action from keywords - BASIC ACTIONS ONLY
            response_lower = response.lower()

            # Map keywords to BASIC actions only!
            if any(
                word in response_lower for word in ["punch", "hit", "attack", "strike"]
            ):
                print(f"🔍 INFERRED FROM 'punch': 9 (LIGHT_PUNCH)")
                return 9  # LIGHT_PUNCH (basic punch)
            elif any(word in response_lower for word in ["kick"]):
                print(f"🔍 INFERRED FROM 'kick': 21 (LIGHT_KICK)")
                return 21  # LIGHT_KICK (basic kick)
            elif any(word in response_lower for word in ["jump", "up"]):
                print(f"🔍 INFERRED FROM 'jump': 1 (UP)")
                return 1  # UP (jump)
            elif any(
                word in response_lower for word in ["right", "forward", "advance"]
            ):
                print(f"🔍 INFERRED FROM 'right': 6 (RIGHT)")
                return 6  # RIGHT (move right)
            elif any(
                word in response_lower for word in ["left", "back", "retreat", "block"]
            ):
                print(f"🔍 INFERRED FROM 'left/block': 3 (LEFT)")
                return 3  # LEFT (move left/block)
            elif any(
                word in response_lower for word in ["crouch", "duck", "down", "low"]
            ):
                print(f"🔍 INFERRED FROM 'crouch': 2 (DOWN)")
                return 2  # DOWN (crouch)

            # BASIC FALLBACK - cycle through basic actions only
            fallback_action = self.aggressive_attacks[self.attack_cycle_index]
            self.attack_cycle_index = (self.attack_cycle_index + 1) % len(
                self.aggressive_attacks
            )
            print(
                f"⚠️ NO KEYWORDS FOUND - CYCLING BASIC FALLBACK: {fallback_action} ({self.action_meanings[fallback_action]})"
            )
            return fallback_action

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
            if len(self.frame_history) >= 4:
                # Use last 4 frames as stack
                for frame_data in self.frame_history[-4:]:
                    frame_stack.append(frame_data["image"])
            else:
                # If we don't have 4 frames yet, pad with current frame
                for frame_data in self.frame_history:
                    frame_stack.append(frame_data["image"])
                # Pad with current frame to reach 4 frames
                while len(frame_stack) < 4:
                    frame_stack.append(image)

            # Use frame stack for temporal understanding
            response = self.query_qwen_vl(frame_stack, prompt)

            # Parse action number from model response
            action = self.parse_action_from_response(response)

            # Prevent repeating the same attack too many times (causes blocking)
            if action == self.last_action:
                self.action_repeat_count += 1
                if self.action_repeat_count > 2:  # If repeating more than 2 times
                    # Force a different attack from our aggressive cycle
                    old_action = action
                    action = self.aggressive_attacks[self.attack_cycle_index]
                    self.attack_cycle_index = (self.attack_cycle_index + 1) % len(
                        self.aggressive_attacks
                    )
                    print(
                        f"🔄 BREAKING REPEAT: {old_action} → {action} ({self.action_meanings[action]})"
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
    model_path: str = "/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-3B-Instruct",  # Function to demo Qwen agent gameplay
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
        "/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-3B-Instruct"
    )  # Create agent with 3B model

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
        description="Qwen Street Fighter Demo"
    )  # Create argument parser
    parser.add_argument(
        "--model",
        type=str,
        default="/home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-3B-Instruct",  # 3B model path argument
        help="SmolVLM model local path",
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
