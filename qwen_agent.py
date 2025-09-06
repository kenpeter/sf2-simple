#!/usr/bin/env python3  # Shebang line to run script with python3 directly
"""
🥊 Qwen-powered Street Fighter 2 Agent
Works with existing wrapper.py without modifications
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

        # Action space - matches discretizer.py
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

        # Action timing - frames each action takes to complete
        self.action_frames = (
            {  # Dictionary mapping action IDs to animation frame durations
                0: 1,  # NO_ACTION - instant response
                1: 3,  # UP - short jump startup frames
                2: 3,  # DOWN - crouch animation frames
                3: 2,  # LEFT - walk animation frames
                6: 2,  # RIGHT - walk animation frames
                7: 5,  # UP_RIGHT - jump animation frames
                4: 5,  # UP_LEFT - jump animation frames
                5: 4,  # DOWN_LEFT - crouch walk animation frames
                8: 4,  # DOWN_RIGHT - crouch walk animation frames
                9: 8,  # LIGHT_PUNCH - fast attack animation frames
                13: 12,  # MEDIUM_PUNCH - medium attack animation frames
                17: 18,  # HEAVY_PUNCH - slow heavy attack animation frames
                21: 10,  # LIGHT_KICK - fast kick animation frames
                26: 15,  # MEDIUM_KICK - medium kick animation frames
                32: 20,  # HEAVY_KICK - slow heavy kick animation frames
                38: 25,  # HADOKEN_RIGHT - fireball animation frames
                39: 22,  # DRAGON_PUNCH_RIGHT - uppercut animation frames
                40: 18,  # HURRICANE_KICK_RIGHT - spinning kick animation frames
                41: 25,  # HADOKEN_LEFT - fireball animation frames
                42: 22,  # DRAGON_PUNCH_LEFT - uppercut animation frames
                43: 18,  # HURRICANE_KICK_LEFT - spinning kick animation frames
            }
        )

        # Game state tracking
        self.action_history = []  # List to store history of past actions
        self.last_features = {}  # Dictionary to store previous game features
        self.frame_counter = 0  # Counter to track current frame number
        self.last_action = 0  # Store the last action taken
        self.last_reasoning = "Initial state"  # Store reasoning for last decision
        self.action_repeat_count = 0  # Count how many times same action repeated
        self.last_distance = 0  # Store previous distance between characters
        self.action_cooldown = 0  # Frames remaining until next action allowed
        self.last_executed_action = 0  # Store the last action that was executed

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

    def create_hybrid_prompt(
        self, features: Dict
    ) -> str:  # Method to create prompt for vision model
        """
        Create optimized prompt for fast gaming decisions

        Args:
            features: Game state features from ta.json

        Returns:
            Optimized prompt for fast gaming response
        """
        # Simplified prompt for faster inference
        prompt = f"""HP: Ken {features['agent_hp']} vs Enemy {features['enemy_hp']} | Dist: {features['distance']}
Actions: 0=WAIT 6=RIGHT 3=LEFT 9=PUNCH 38=FIREBALL 39=UPPERCUT
Choose action number:"""  # Simplified prompt for faster processing

        return prompt  # Return the optimized prompt string

    def query_qwen_vl(
        self, image: Image.Image, prompt: str
    ) -> str:  # Method to query Qwen2.5-VL model
        """
        Query Qwen2.5-VL model with image and prompt

        Args:
            image: Game frame as PIL Image
            prompt: Text prompt for analysis

        Returns:
            Model's response containing reasoning and action
        """
        # Create messages for Qwen2.5-VL format
        messages = [  # Create message list in chat format
            {  # User message containing image and text
                "role": "user",  # Set role as user
                "content": [  # Content list with image and text
                    {"type": "image", "image": image},  # Image component
                    {"type": "text", "text": prompt},  # Text prompt component
                ],
            }
        ]

        # Simplified approach - directly process text and image like the working example
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Debug the inputs being generated
        inputs = self.processor(
            text=text_input,
            images=image,
            return_tensors="pt"
        )
        
        # DEBUG: Print what we're actually sending to the model
        print(f"\n🔧 DEBUG INPUT PROCESSING:")
        print(f"Text input length: {len(text_input)}")
        print(f"Image type: {type(image)}")
        print(f"Image size: {image.size}")
        print(f"Input keys: {list(inputs.keys())}")
        if 'pixel_values' in inputs:
            print(f"✅ pixel_values shape: {inputs['pixel_values'].shape}")
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
                max_new_tokens=100,  # More tokens for detailed debugging analysis
                do_sample=False,  # Use greedy decoding (deterministic)
                pad_token_id=self.processor.tokenizer.pad_token_id,  # Set padding token
                num_beams=1,  # Single beam for maximum speed
                use_cache=True,  # Enable KV cache for speed
            )

        # Decode response
        response = self.processor.decode(
            outputs[0], skip_special_tokens=True
        )  # Decode tokens to text
        # Extract just the action part after the prompt
        if "Action:" in response:  # Check if Action: marker exists
            response = response.split("Action:")[
                -1
            ].strip()  # Extract text after Action: marker
        return response.strip()  # Return cleaned response

    def extract_game_state_from_vision(self, image: Image.Image) -> Dict:
        """
        Extract detailed game state information using pure Qwen2.5-VL vision analysis
        
        Args:
            image: Game frame as PIL Image
            
        Returns:
            Dictionary containing extracted game state information
        """
        # First check if frame has content, then analyze if it does
        analysis_prompt = '''Describe this image briefly. What do you see? Is it mostly black/empty or does it contain game content?'''
        
        try:
            # Get structured analysis from Qwen2.5-VL
            response = self.query_qwen_vl(image, analysis_prompt)
            
            # LOG: Print what the model actually sees and understands
            import time
            timestamp = time.strftime("%H:%M:%S")
            print(f"\n🔍 VISION MODEL ANALYSIS (Frame {self.frame_counter} at {timestamp}):")
            print(f"PROMPT: {analysis_prompt}")
            print(f"\n🤖 MODEL RESPONSE:")
            print(f"{response}")
            print("-" * 80)
            
            # Try to extract JSON from response
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    game_state = json.loads(json_str)
                    print(f"✅ PARSED JSON: {game_state}")
                    return game_state
                except json.JSONDecodeError as je:
                    print(f"❌ JSON PARSE ERROR: {je}")
                    pass
            
            # Fallback: basic state analysis
            fallback_state = self._fallback_state_analysis(response)
            print(f"🔄 FALLBACK STATE: {fallback_state}")
            return fallback_state
            
        except Exception as e:
            print(f"❌ VISION ANALYSIS ERROR: {e}")
            default_state = self._default_game_state()
            print(f"⚠️ DEFAULT STATE: {default_state}")
            return default_state
    
    def _fallback_state_analysis(self, response: str) -> Dict:
        """Fallback method to extract basic game state from text response"""
        # Basic text parsing for key information
        state = self._default_game_state()
        
        response_lower = response.lower()
        
        # Detect health mentions
        if "low health" in response_lower or "critical" in response_lower:
            state["action_context"]["recent_hit"] = True
            
        # Detect projectiles
        if "fireball" in response_lower or "projectile" in response_lower:
            state["game_elements"]["projectiles_present"] = True
            
        # Detect action intensity
        if any(word in response_lower for word in ["intense", "combo", "attacking", "fighting"]):
            state["action_context"]["intensity_level"] = "intense"
        elif any(word in response_lower for word in ["calm", "neutral", "waiting"]):
            state["action_context"]["intensity_level"] = "calm"
        else:
            state["action_context"]["intensity_level"] = "moderate"
            
        return state
    
    def _default_game_state(self) -> Dict:
        """Default game state structure"""
        return {
            "characters": {
                "player1": {
                    "name": "unknown",
                    "position": "left",
                    "health_percentage": 50,
                    "stance": "standing",
                    "special_state": "normal"
                },
                "player2": {
                    "name": "unknown",
                    "position": "right", 
                    "health_percentage": 50,
                    "stance": "standing",
                    "special_state": "normal"
                }
            },
            "game_elements": {
                "projectiles_present": False,
                "projectile_locations": [],
                "special_effects": [],
                "round_timer": "unknown",
                "distance_between_fighters": "medium"
            },
            "action_context": {
                "intensity_level": "moderate",
                "primary_action_area": "center",
                "recent_hit": False,
                "combo_in_progress": False
            }
        }

    def vision_based_decision_making(
        self, features: Dict, image: Image.Image, prompt: str
    ) -> Tuple[int, str]:  # Pure vision-based AI decision making with Qwen2.5-VL
        """
        Enhanced decision making using pure Qwen2.5-VL vision analysis

        Args:
            features: Game state features (legacy, for compatibility)
            image: Current game frame for visual analysis
            prompt: Base prompt for Qwen2.5-VL

        Returns:
            Tuple of (action_number, reasoning_text)
        """
        # Extract detailed game state using pure vision
        game_state = self.extract_game_state_from_vision(image)
        
        # Dynamic aggressive action selection with variety and combos
        distance = game_state.get('game_elements', {}).get('distance_between_fighters', 'medium')
        projectiles = game_state.get('game_elements', {}).get('projectiles_present', False)
        my_health = game_state.get('characters', {}).get('player1', {}).get('health_percentage', 50)
        enemy_health = game_state.get('characters', {}).get('player2', {}).get('health_percentage', 50)
        
        # Add frame-based variation to prevent loops
        action_cycle = (self.frame_counter // 6) % 4  # Change strategy every 6 frames, 4 different patterns
        
        # LOG: Show what the vision system extracted
        print(f"\n🎯 DECISION MAKING (Frame {self.frame_counter}):")
        print(f"Distance: {distance}")
        print(f"Projectiles: {projectiles}")
        print(f"My Health: {my_health}%")
        print(f"Enemy Health: {enemy_health}%")
        print(f"Action Cycle: {action_cycle}")
        
        # More aggressive and varied action selection
        if projectiles and distance == 'far':
            # Multiple ways to handle projectiles
            if action_cycle == 0:
                action = 1  # JUMP over
                reasoning = "Vision: Projectiles - jumping over"
            elif action_cycle == 1:
                action = 2  # DUCK under
                reasoning = "Vision: Projectiles - ducking under"
            elif action_cycle == 2:
                action = 38  # Counter with hadoken
                reasoning = "Vision: Projectiles - counter hadoken"
            else:
                action = 6  # Move forward aggressively
                reasoning = "Vision: Projectiles - advancing through"
                
        elif distance == 'far':
            # Long range - mix of approaches
            if my_health < enemy_health:
                # Behind in health - be aggressive
                if action_cycle % 2 == 0:
                    action = 6  # Move forward
                    reasoning = "Vision: Far range, behind in health - advancing"
                else:
                    action = 38  # Hadoken while advancing
                    reasoning = "Vision: Far range, behind in health - hadoken pressure"
            else:
                # Ahead or even - control space
                if action_cycle == 0:
                    action = 38  # Hadoken
                    reasoning = "Vision: Far range - space control hadoken"
                elif action_cycle == 1:
                    action = 6   # Move forward
                    reasoning = "Vision: Far range - advancing"
                elif action_cycle == 2:
                    action = 1   # Jump forward
                    reasoning = "Vision: Far range - jump approach"
                else:
                    action = 17  # Heavy punch (for spacing)
                    reasoning = "Vision: Far range - heavy spacing"
                    
        elif distance == 'close':
            # Close range - combo sequences and mix-ups
            if action_cycle == 0:
                # Combo sequence 1: Heavy punch
                action = 17  # HEAVY_PUNCH
                reasoning = "Vision: Close range - heavy combo starter"
            elif action_cycle == 1:
                # Combo sequence 2: Dragon punch
                action = 39  # DRAGON_PUNCH
                reasoning = "Vision: Close range - anti-air dragon punch"
            elif action_cycle == 2:
                # Combo sequence 3: Hurricane kick
                action = 40  # HURRICANE_KICK
                reasoning = "Vision: Close range - hurricane kick pressure"
            else:
                # Combo sequence 4: Throw attempt
                action = 21  # MEDIUM_KICK (closest to throw)
                reasoning = "Vision: Close range - throw attempt"
                
        elif distance == 'medium':
            # Medium range - varied approaches instead of just jumping
            if action_cycle == 0:
                action = 1   # JUMP attack
                reasoning = "Vision: Medium range - jump attack vs M. Bison"
            elif action_cycle == 1:
                action = 38  # Hadoken
                reasoning = "Vision: Medium range - hadoken pressure"
            elif action_cycle == 2:
                action = 6   # Move forward
                reasoning = "Vision: Medium range - advancing"
            else:
                action = 17  # Heavy punch
                reasoning = "Vision: Medium range - heavy poke"
        else:
            # Fallback with more variety
            import random
            aggressive_actions = [6, 17, 38, 39, 40, 1, 21, 13]  # Mix of movement, attacks, specials
            action = random.choice(aggressive_actions)
            reasoning = f"Vision: Dynamic fallback - action {action}"
        
        # LOG: Final decision
        action_name = self.action_meanings[action] if action < len(self.action_meanings) else 'UNKNOWN'
        print(f"✅ FINAL DECISION: Action {action} ({action_name})")
        print(f"Reasoning: {reasoning}")
        print("=" * 60)
        
        return action, reasoning


    def parse_action_from_response(self, response: str) -> int:
        """
        Parse the action number from Qwen's response

        Args:
            response: Raw response from Qwen

        Returns:
            Action number (0-43), defaults to 0 if parsing fails
        """
        try:
            # Look for "Action: X" pattern
            action_match = re.search(r"Action:\s*(\d+)", response, re.IGNORECASE)
            if action_match:
                action = int(action_match.group(1))
                if 0 <= action < self.num_actions:
                    return action

            # Fallback: look for any number in valid range at start of line
            lines = response.split("\n")
            for line in lines:
                numbers = re.findall(r"\b(\d+)\b", line)
                for num_str in numbers:
                    num = int(num_str)
                    if 0 <= num < self.num_actions:
                        return num

            # Default fallback
            return 0

        except Exception as e:
            print(f"⚠️  Action parsing failed: {e}")
            return 0  # Default to no action

    def get_action(
        self, observation, info: Dict, verbose: bool = False
    ) -> Tuple[int, str]:
        """
        Get action decision from Qwen2.5-VL based on visual frame analysis
        Processes every 30th frame to avoid black frames and ensure game stability

        Args:
            observation: Game frame (numpy array or PIL Image)
            info: Game state info from environment
            verbose: Whether to print reasoning

        Returns:
            Tuple of (action_number, reasoning_text)
        """
        self.frame_counter += 1

        # Process every 30th frame to give game plenty of time to initialize and render
        if (
            self.frame_counter % 30 == 0 or self.frame_counter == 60
        ):  # Process every 30th frame, starting after initialization
            # Vision analysis every 30th frame for better frame quality
            image = self.capture_game_frame(observation)
            features = self.extract_game_features(info)
            prompt = self.create_hybrid_prompt(features)

            # Enhanced decision making: OpenCV analysis + SmolVLM reasoning
            action, response = self.vision_based_decision_making(features, image, prompt)

            # Cache the new decision
            self.last_action = action
            self.last_reasoning = response

        else:
            # Use cached action from last model inference (every 30th frame caching)
            action = self.last_action
            response = self.last_reasoning + " (cached)"

        # Update action history
        action_name = self.action_meanings[action]
        self.action_history.append(action_name)

        # Keep history manageable
        if len(self.action_history) > 20:
            self.action_history = self.action_history[-20:]

        if verbose:
            model_status = (
                "NEW"
                if (self.frame_counter % 5 == 0 or self.frame_counter == 1)
                else "CACHED"
            )
            print(f"\n🚀 Qwen2.5-VL INT4 Decision ({model_status}):")
            print(f"Frame: {self.frame_counter}")
            print(f"Action: {action} ({action_name})")
            print(f"Reasoning: {response}")

        return action, response

    def reset(self):  # Method to reset agent state
        """Reset the agent state"""
        self.action_history = []  # Clear action history list
        self.last_features = {}  # Clear previous game features
        self.frame_counter = 0  # Reset frame counter to zero
        self.last_action = 0  # Reset last action to NO_ACTION
        self.last_reasoning = "Reset state"  # Reset reasoning to initial state
        self.action_repeat_count = 0  # Reset action repeat counter
        self.last_distance = 0  # Reset last distance measurement
        self.action_cooldown = 0  # Reset action cooldown timer
        self.last_executed_action = 0  # Reset last executed action


# Test script
if __name__ == "__main__":  # Check if script is run directly
    print("🥊 Testing Qwen Street Fighter Agent")  # Print test header

    # Create mock info for testing
    mock_info = {  # Dictionary with mock game state data
        "agent_hp": 150,  # Mock agent health points
        "agent_x": 200,  # Mock agent x-coordinate
        "agent_y": 100,  # Mock agent y-coordinate
        "enemy_hp": 120,  # Mock enemy health points
        "enemy_x": 300,  # Mock enemy x-coordinate
        "enemy_y": 100,  # Mock enemy y-coordinate
        "agent_status": 0,  # Mock agent animation status
        "enemy_status": 5,  # Mock enemy animation status
    }

    try:  # Try to run test with error handling
        # Create agent (will download model if needed)
        agent = QwenStreetFighterAgent()  # Initialize agent with default model path

        # Test action selection with dummy frame
        dummy_frame = np.zeros((224, 320, 3), dtype=np.uint8)  # Create black RGB frame
        action, reasoning = agent.get_action(
            dummy_frame, mock_info, verbose=True
        )  # Test action selection

        print(
            f"\n✅ Test completed! Chosen action: {action}"
        )  # Print successful test result

    except Exception as e:  # Catch any exceptions
        print(f"❌ Test failed: {e}")  # Print error message
        print(
            "Note: You'll need to download Qwen weights separately"
        )  # Print note about model requirements
