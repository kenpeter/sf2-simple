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
        
        # Game state tracking
        self.action_history = []
        self.last_features = {}
        
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
        prompt = f"""Street Fighter 2: You control Ken. Analyze the visual frame AND the game data below.

GAME DATA:
Ken HP: {features['agent_hp']} | Enemy HP: {features['enemy_hp']} | Distance: {features['distance']}
Ken position: ({features['agent_x']},{features['agent_y']}) | Enemy: ({features['enemy_x']},{features['enemy_y']})
Score: {features['score']} | Time: {features['round_countdown']} | Facing: {features['facing']}

Look at the visual frame to understand the fight situation. Use both visual and data info to choose the best action.

COMBO STRATEGY:
- Close range: Heavy attacks into special moves
- Medium range: Hadoken fireballs to control space
- Anti-air: Dragon Punch when enemy jumps
- Pressure: Hurricane Kick for mobility and attacks

ACTIONS (choose best for situation):
Move: 0=NONE 1=UP 2=DOWN 3=LEFT 6=RIGHT 7=UP_RIGHT 4=UP_LEFT 5=DOWN_LEFT 8=DOWN_RIGHT
Attack: 9=L_PUNCH 13=M_PUNCH 17=H_PUNCH 21=L_KICK 26=M_KICK 32=H_KICK
SPECIAL MOVES: 38=HADOKEN_R 39=DRAGON_PUNCH_R 40=HURRICANE_KICK_R 41=HADOKEN_L 42=DRAGON_PUNCH_L 43=HURRICANE_KICK_L

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
        # Create messages for SmolVLM
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process with SmolVLM processor (simpler than Qwen2.5-VL)
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response with minimal tokens for speed
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8,   # Very short for speed
                do_sample=False,    # Greedy decoding
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        # Extract just the action part
        if "Action:" in response:
            response = response.split("Action:")[-1].strip()
        return response.strip()
    
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
        
        Args:
            observation: Game frame (numpy array or PIL Image)
            info: Game state info from environment
            verbose: Whether to print reasoning
            
        Returns:
            Tuple of (action_number, reasoning_text)
        """
        # Real-time SmolVLM analysis every frame
        image = self.capture_game_frame(observation)
        features = self.extract_game_features(info)
        prompt = self.create_hybrid_prompt(features)
        
        try:
            response = self.query_smolvlm(image, prompt)
            action = self.parse_action_from_response(response)
            
        except Exception:
            # Fast fallback using game features
            import random
            distance = features['distance']
            agent_hp = features['agent_hp']
            enemy_hp = features['enemy_hp']
            
            if agent_hp < enemy_hp and distance > 150:
                action = random.choice([38, 41, 39, 42])  # Defensive projectiles and dragon punch
            elif distance < 80:
                action = random.choice([9, 13, 17, 21, 26, 32, 39, 42, 40, 43])  # Close combat with specials
            elif distance < 150:
                action = random.choice([38, 41, 17, 32, 40, 43])  # Medium range with hurricane kicks
            else:
                action = random.choice([6, 3, 38, 41, 40, 43])  # Movement, projectiles, hurricane kicks
            
            response = f"Fallback: d={distance}, hp={agent_hp}/{enemy_hp}"
        
        # Update action history
        action_name = self.action_meanings[action]
        self.action_history.append(action_name)
        
        # Keep history manageable
        if len(self.action_history) > 20:
            self.action_history = self.action_history[-20:]
        
        if verbose:
            print(f"\n🧠 SmolVLM Real-Time Decision:")
            print(f"Action: {action} ({action_name})")
            print(f"Reasoning: {response}")
        
        return action, response
    
    def reset(self):
        """Reset the agent state"""
        self.action_history = []
        self.last_features = {}


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