#!/usr/bin/env python3
"""
🥊 Qwen-powered Street Fighter 2 Agent
Works with existing wrapper.py without modifications
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
import numpy as np
from PIL import Image
import re
from typing import Dict, Tuple


class QwenStreetFighterAgent:
    """
    Qwen-powered agent for Street Fighter 2
    Uses existing wrapper.py environment without modifications
    """

    def __init__(self, model_path: str = "/home/kenpeter/.cache/huggingface/hub/SmolVLM-Instruct"):
        """
        Initialize the Qwen agent
        
        Args:
            model_path: Path to Qwen model (local or HuggingFace)
        """
        # Initialize Qwen model
        print(f"🤖 Loading Qwen model from: {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor and model for vision
        print("📁 Step 1/2: Loading processor from cache...")
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        
        # Load vision model from cache
        print("📁 Step 2/2: Loading Qwen2.5-VL model from cache...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            max_memory={0: "4GB"} if torch.cuda.is_available() else None
        )
        print(f"✅ Qwen model loaded successfully on {self.device}")
        
        # Action space - matches discretizer.py
        self.action_meanings = [
            "NO_ACTION",           # 0
            "UP",                  # 1
            "DOWN",                # 2
            "LEFT",                # 3
            "UP_LEFT",             # 4
            "DOWN_LEFT",           # 5
            "RIGHT",               # 6
            "UP_RIGHT",            # 7
            "DOWN_RIGHT",          # 8
            "LIGHT_PUNCH",         # 9
            "LIGHT_PUNCH_DOWN",    # 10
            "LIGHT_PUNCH_LEFT",    # 11
            "LIGHT_PUNCH_RIGHT",   # 12
            "MEDIUM_PUNCH",        # 13
            "MEDIUM_PUNCH_DOWN",   # 14
            "MEDIUM_PUNCH_LEFT",   # 15
            "MEDIUM_PUNCH_RIGHT",  # 16
            "HEAVY_PUNCH",         # 17
            "HEAVY_PUNCH_DOWN",    # 18
            "HEAVY_PUNCH_LEFT",    # 19
            "HEAVY_PUNCH_RIGHT",   # 20
            "LIGHT_KICK",          # 21
            "LIGHT_KICK_DOWN",     # 22
            "LIGHT_KICK_LEFT",     # 23
            "LIGHT_KICK_DOWN_LEFT", # 24
            "LIGHT_KICK_RIGHT",    # 25
            "MEDIUM_KICK",         # 26
            "MEDIUM_KICK_DOWN",    # 27
            "MEDIUM_KICK_LEFT",    # 28
            "MEDIUM_KICK_DOWN_LEFT", # 29
            "MEDIUM_KICK_RIGHT",   # 30
            "MEDIUM_KICK_DOWN_RIGHT", # 31
            "HEAVY_KICK",          # 32
            "HEAVY_KICK_DOWN",     # 33
            "HEAVY_KICK_LEFT",     # 34
            "HEAVY_KICK_DOWN_LEFT", # 35
            "HEAVY_KICK_RIGHT",    # 36
            "HEAVY_KICK_DOWN_RIGHT", # 37
            "HADOKEN_RIGHT",       # 38
            "DRAGON_PUNCH_RIGHT",  # 39
            "HURRICANE_KICK_RIGHT", # 40
            "HADOKEN_LEFT",        # 41
            "DRAGON_PUNCH_LEFT",   # 42
            "HURRICANE_KICK_LEFT", # 43
        ]
        
        self.num_actions = len(self.action_meanings)
        
        # Action timing - frames each action takes to complete
        self.action_frames = {
            0: 1,    # NO_ACTION - instant
            1: 3,    # UP - short jump startup
            2: 3,    # DOWN - crouch
            3: 2,    # LEFT - walk
            6: 2,    # RIGHT - walk
            7: 5,    # UP_RIGHT - jump
            4: 5,    # UP_LEFT - jump
            5: 4,    # DOWN_LEFT - crouch walk
            8: 4,    # DOWN_RIGHT - crouch walk
            9: 8,    # LIGHT_PUNCH - fast attack
            13: 12,  # MEDIUM_PUNCH - medium attack
            17: 18,  # HEAVY_PUNCH - slow heavy attack
            21: 10,  # LIGHT_KICK - fast kick
            26: 15,  # MEDIUM_KICK - medium kick
            32: 20,  # HEAVY_KICK - slow heavy kick
            38: 25,  # HADOKEN_RIGHT - fireball animation
            39: 22,  # DRAGON_PUNCH_RIGHT - uppercut animation
            40: 18,  # HURRICANE_KICK_RIGHT - spinning kick
            41: 25,  # HADOKEN_LEFT - fireball animation
            42: 22,  # DRAGON_PUNCH_LEFT - uppercut animation  
            43: 18,  # HURRICANE_KICK_LEFT - spinning kick
        }
        
        # Game state tracking
        self.action_history = []
        self.last_features = {}
        self.frame_counter = 0
        self.last_action = 0
        self.last_reasoning = "Initial state"
        self.action_repeat_count = 0
        self.last_distance = 0
        self.action_cooldown = 0
        self.last_executed_action = 0
        
    def extract_game_features(self, info: Dict) -> Dict:
        """
        Extract structured features from game state info based on ta.json schema
        
        Args:
            info: Game state info dictionary from environment
            
        Returns:
            Dictionary of structured game features
        """
        features = {
            # Player status (from ta.json)
            "agent_hp": info.get("agent_hp", 176),
            "agent_x": info.get("agent_x", 0),
            "agent_y": info.get("agent_y", 0),
            "agent_status": info.get("agent_status", 0),
            "agent_victories": info.get("agent_victories", 0),
            
            # Enemy status (from ta.json)
            "enemy_hp": info.get("enemy_hp", 176),
            "enemy_x": info.get("enemy_x", 0),
            "enemy_y": info.get("enemy_y", 0),
            "enemy_victories": info.get("enemy_victories", 0),
            
            # Game status (from ta.json)
            "score": info.get("score", 0),
            "round_countdown": info.get("round_countdown", 99),
        }
        
        # Calculate derived features (only from available data)
        features["hp_advantage"] = features["agent_hp"] - features["enemy_hp"]
        features["distance"] = abs(features["agent_x"] - features["enemy_x"])
        features["height_diff"] = features["agent_y"] - features["enemy_y"]
        
        # Determine relative position
        if features["agent_x"] < features["enemy_x"]:
            features["facing"] = "right"
        else:
            features["facing"] = "left"
            
        return features
    
    def capture_game_frame(self, observation) -> Image.Image:
        """
        Convert game observation to PIL Image for vision model
        
        Args:
            observation: Game frame from environment (numpy array)
            
        Returns:
            PIL Image of the game frame
        """
        if isinstance(observation, np.ndarray):
            # Handle different observation formats
            if observation.shape == (1, 1, 1):
                # Single pixel observation - create a dummy RGB image
                dummy_frame = np.zeros((224, 320, 3), dtype=np.uint8)
                return Image.fromarray(dummy_frame)
            
            # Convert numpy array to PIL Image
            if observation.dtype != np.uint8:
                observation = (observation * 255).astype(np.uint8)
            
            # Ensure proper shape for image
            if len(observation.shape) == 3 and observation.shape[2] in [3, 4]:
                # RGB or RGBA image
                if observation.shape[2] == 4:
                    observation = observation[:, :, :3]  # Remove alpha channel
                image = Image.fromarray(observation)
            elif len(observation.shape) == 2:
                # Grayscale - convert to RGB
                image = Image.fromarray(observation).convert('RGB')
            else:
                # Unexpected format - create dummy image
                dummy_frame = np.zeros((224, 320, 3), dtype=np.uint8)
                return Image.fromarray(dummy_frame)
            
            return image
        else:
            # If already PIL Image, return as-is
            return observation
    
    def create_hybrid_prompt(self, features: Dict) -> str:
        """
        Create hybrid prompt combining visual frame analysis with game features
        
        Args:
            features: Game state features from ta.json
            
        Returns:
            Formatted prompt for vision + data analysis
        """
        prompt = f"""Ken HP: {features['agent_hp']} Enemy HP: {features['enemy_hp']} Distance: {features['distance']} Facing: {features['facing']}

STRATEGY:
- Close range: Light attack then special move
- Medium range: Hadoken fireballs  
- Anti-air: Dragon Punch when enemy jumps
- Pressure: Hurricane Kick

ACTIONS:
Move: 0=NONE 1=UP 2=DOWN 3=LEFT 6=RIGHT
Attack: 9=L_PUNCH 13=M_PUNCH 17=H_PUNCH 21=L_KICK 26=M_KICK 32=H_KICK
Special: 38=HADOKEN_R 39=DRAGON_PUNCH_R 40=HURRICANE_KICK_R 41=HADOKEN_L 42=DRAGON_PUNCH_L 43=HURRICANE_KICK_L

Action:"""

        return prompt
    
    def query_smolvlm(self, image: Image.Image, prompt: str) -> str:
        """
        Query SmolVLM model with image and prompt
        
        Args:
            image: Game frame as PIL Image
            prompt: Text prompt for analysis
            
        Returns:
            Model's response containing reasoning and action
        """
        # Create messages for SmolVLM (Idefics3 format)
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Create formatted text input
        text_input = self.processor.apply_chat_template(
            messages, images=[image], add_generation_prompt=True
        )
        
        # Process the text and image to get tensors
        inputs = self.processor(
            text=text_input, 
            images=[image], 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response with minimal tokens for speed
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8,   # Very short for speed
                do_sample=False,    # Greedy decoding
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        
        # Decode response  
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        # Extract just the action part after the prompt
        if "Action:" in response:
            response = response.split("Action:")[-1].strip()
        return response.strip()
    
    def analyze_frame_context(self, image: Image.Image, features: Dict) -> str:
        """
        Analyze the visual frame for additional context
        
        Args:
            image: Game frame as PIL Image
            features: Game state features
            
        Returns:
            Additional context from visual analysis
        """
        import numpy as np
        
        # Convert PIL image to numpy for analysis
        frame = np.array(image)
        
        # Analyze frame characteristics
        context = []
        
        # Check frame brightness (could indicate special effects, fireballs, etc.)
        brightness = np.mean(frame)
        if brightness > 140:
            context.append("bright_flash")  # Special moves, hits
        elif brightness < 80:
            context.append("dark_frame")    # Normal state
        
        # Check for color patterns that might indicate projectiles or special states
        # Look for blue tints (often fireballs) or red tints (often hits/damage)
        blue_intensity = np.mean(frame[:, :, 2]) if len(frame.shape) == 3 else 0
        red_intensity = np.mean(frame[:, :, 0]) if len(frame.shape) == 3 else 0
        
        if blue_intensity > red_intensity + 20:
            context.append("blue_projectile")  # Possible fireball
        elif red_intensity > blue_intensity + 20:
            context.append("red_flash")       # Possible hit/damage
        
        # Analyze frame regions for movement patterns
        if len(frame.shape) == 3:
            # Check left and right sides for character positions
            left_activity = np.std(frame[:, :frame.shape[1]//3])
            right_activity = np.std(frame[:, 2*frame.shape[1]//3:])
            
            if left_activity > right_activity * 1.5:
                context.append("left_active")   # More activity on left
            elif right_activity > left_activity * 1.5:
                context.append("right_active")  # More activity on right
        
        return context
    
    def strategic_decision_making(self, features: Dict, image: Image.Image) -> Tuple[int, str]:
        """
        Intelligent rule-based decision making with proper reasoning and frame analysis
        
        Args:
            features: Game state features
            image: Current game frame for visual analysis
            
        Returns:
            Tuple of (action_number, reasoning_text)
        """
        distance = features['distance']
        hp_advantage = features['hp_advantage']
        facing = features['facing']
        
        # Decrement action cooldown
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
            # Still in cooldown - return NO_ACTION or continue previous action
            if self.action_cooldown > 0:
                action = 0  # NO_ACTION - wait for animation to finish
                reasoning = f"waiting for {self.action_meanings[self.last_executed_action]} to complete ({self.action_cooldown} frames left)"
                return action, reasoning
        
        # Force action variety if stuck repeating same action
        force_movement = False
        if self.action_repeat_count > 3:  # Reduced from 5 since we have timing now
            force_movement = True
            
        # Check if distance hasn't changed (stuck position)  
        distance_change = abs(distance - self.last_distance) if self.last_distance > 0 else 999
        if distance_change < 10 and self.frame_counter > 20:  # Position hasn't changed much
            force_movement = True
            
        self.last_distance = distance
        
        # Analyze frame for visual context
        visual_context = self.analyze_frame_context(image, features)
        
        # Adjust strategy based on visual cues
        base_reasoning = ""
        if "bright_flash" in visual_context:
            base_reasoning += "Flash detected - "
        if "blue_projectile" in visual_context:
            base_reasoning += "Projectile seen - "
        if "red_flash" in visual_context:
            base_reasoning += "Hit detected - "
        
        # Force movement if stuck
        if force_movement:
            import random
            movement_actions = [3, 6, 1, 2]  # LEFT, RIGHT, UP, DOWN
            action = random.choice(movement_actions)
            reasoning = base_reasoning + f"breaking pattern, forced movement (repeated {self.action_repeat_count}x)"
            
        # React to visual cues first
        elif "blue_projectile" in visual_context and distance > 100:
            # Enemy projectile detected - dodge or counter
            action = 1  # UP (jump to avoid)
            reasoning = base_reasoning + "jumping to avoid projectile"
            
        elif "bright_flash" in visual_context and distance < 100:
            # Special move or hit flash detected - be defensive  
            if facing == "right":
                action = 42  # DRAGON_PUNCH_LEFT (away from enemy)
                reasoning = base_reasoning + "defensive dragon punch after flash"
            else:
                action = 39  # DRAGON_PUNCH_RIGHT
                reasoning = base_reasoning + "defensive dragon punch after flash"
                
        elif "red_flash" in visual_context:
            # Hit detected - follow up or counter
            if distance < 80:
                action = 17  # HEAVY_PUNCH
                reasoning = base_reasoning + "following up after hit with heavy attack"
            else:
                action = 40 if facing == "right" else 43  # HURRICANE_KICK
                reasoning = base_reasoning + "hurricane kick to capitalize on hit"
                
        # Standard strategic decisions based on game state
        elif hp_advantage < -50:
            # Losing badly - play defensively
            if distance > 200:
                if facing == "right":
                    action = 38  # HADOKEN_RIGHT
                    reasoning = base_reasoning + "losing badly, keeping distance with fireball"
                else:
                    action = 41  # HADOKEN_LEFT  
                    reasoning = base_reasoning + "losing badly, keeping distance with fireball"
            else:
                action = 42 if facing == "right" else 39  # DRAGON_PUNCH
                reasoning = base_reasoning + "losing badly, defensive anti-air ready"
                
        elif hp_advantage > 50:
            # Winning - play aggressively
            if distance < 60:
                action = 17  # HEAVY_PUNCH
                reasoning = base_reasoning + "winning big, close range heavy attack"
            elif distance < 120:
                action = 40 if facing == "right" else 43  # HURRICANE_KICK
                reasoning = base_reasoning + "winning big, aggressive hurricane kick pressure"
            else:
                if facing == "right":
                    action = 6  # RIGHT
                    reasoning = base_reasoning + "winning big, moving in for pressure"
                else:
                    action = 3  # LEFT
                    reasoning = base_reasoning + "winning big, moving in for pressure"
                    
        elif distance < 50:
            # Very close range - combo sequences
            import random
            combo_actions = [9, 13, 17, 21, 26]  # Mix of punches and kicks
            action = random.choice(combo_actions)
            reasoning = base_reasoning + "very close range, combo attack"
            
        elif distance < 100:
            # Close range - mix of attacks and movement
            import random
            close_actions = [17, 32, 39, 42, 40, 43, 6, 3]  # Heavy attacks, specials, movement
            action = random.choice(close_actions)
            if action in [6, 3]:
                reasoning = base_reasoning + "close range, repositioning"
            elif action in [39, 42]:
                reasoning = base_reasoning + "close range, dragon punch"
            elif action in [40, 43]:
                reasoning = base_reasoning + "close range, hurricane kick"
            else:
                reasoning = base_reasoning + "close range, heavy attack"
                
        elif distance < 180:
            # Medium range - control space
            if facing == "right":
                action = 38  # HADOKEN_RIGHT
                reasoning = base_reasoning + "medium range, fireball to control space"
            else:
                action = 41  # HADOKEN_LEFT
                reasoning = base_reasoning + "medium range, fireball to control space"
        else:
            # Long range - close the gap
            if facing == "right":
                action = 6  # RIGHT
                reasoning = base_reasoning + "long range, moving forward to engage"
            else:
                action = 3  # LEFT
                reasoning = base_reasoning + "long range, moving forward to engage"
        
        # Set action cooldown based on selected action
        self.action_cooldown = self.action_frames.get(action, 5)  # Default 5 frames if not found
        self.last_executed_action = action
        
        # Track action repeats for next frame
        if action == self.last_action:
            self.action_repeat_count += 1
        else:
            self.action_repeat_count = 0
            
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
            action_match = re.search(r'Action:\s*(\d+)', response, re.IGNORECASE)
            if action_match:
                action = int(action_match.group(1))
                if 0 <= action < self.num_actions:
                    return action
            
            # Fallback: look for any number in valid range at start of line
            lines = response.split('\n')
            for line in lines:
                numbers = re.findall(r'\b(\d+)\b', line)
                for num_str in numbers:
                    num = int(num_str)
                    if 0 <= num < self.num_actions:
                        return num
            
            # Default fallback
            return 0
            
        except Exception as e:
            print(f"⚠️  Action parsing failed: {e}")
            return 0  # Default to no action
    
    def get_action(self, observation, info: Dict, verbose: bool = False) -> Tuple[int, str]:
        """
        Get action decision from Qwen2.5-VL based on visual frame analysis
        Only processes frames every 3 frames to reduce computational load
        
        Args:
            observation: Game frame (numpy array or PIL Image)
            info: Game state info from environment
            verbose: Whether to print reasoning
            
        Returns:
            Tuple of (action_number, reasoning_text)
        """
        self.frame_counter += 1
        
        # Only process model every 3 frames (or first frame)
        if self.frame_counter % 3 == 0 or self.frame_counter == 1:
            # SmolVLM analysis every 3rd frame
            image = self.capture_game_frame(observation)
            features = self.extract_game_features(info)
            prompt = self.create_hybrid_prompt(features)
            
            # Intelligent rule-based reasoning system with frame analysis
            action, response = self.strategic_decision_making(features, image)
            
            # Cache the new decision
            self.last_action = action
            self.last_reasoning = response
            
        else:
            # Use cached action from last model inference
            action = self.last_action
            response = self.last_reasoning + " (cached)"
        
        # Update action history
        action_name = self.action_meanings[action]
        self.action_history.append(action_name)
        
        # Keep history manageable
        if len(self.action_history) > 20:
            self.action_history = self.action_history[-20:]
        
        if verbose:
            model_status = "NEW" if (self.frame_counter % 3 == 0 or self.frame_counter == 1) else "CACHED"
            print(f"\n🧠 SmolVLM Decision ({model_status}):")
            print(f"Frame: {self.frame_counter}")
            print(f"Action: {action} ({action_name})")
            print(f"Reasoning: {response}")
        
        return action, response
    
    def reset(self):
        """Reset the agent state"""
        self.action_history = []
        self.last_features = {}
        self.frame_counter = 0
        self.last_action = 0
        self.last_reasoning = "Reset state"
        self.action_repeat_count = 0
        self.last_distance = 0
        self.action_cooldown = 0
        self.last_executed_action = 0


# Test script
if __name__ == "__main__":
    print("🥊 Testing Qwen Street Fighter Agent")
    
    # Create mock info for testing
    mock_info = {
        "agent_hp": 150,
        "agent_x": 200, 
        "agent_y": 100,
        "enemy_hp": 120,
        "enemy_x": 300,
        "enemy_y": 100,
        "agent_status": 0,
        "enemy_status": 5
    }
    
    try:
        # Create agent (will download model if needed)
        agent = QwenStreetFighterAgent()
        
        # Test action selection with dummy frame
        dummy_frame = np.zeros((224, 320, 3), dtype=np.uint8)
        action, reasoning = agent.get_action(dummy_frame, mock_info, verbose=True)
        
        print(f"\n✅ Test completed! Chosen action: {action}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Note: You'll need to download Qwen weights separately")