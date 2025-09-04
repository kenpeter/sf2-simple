#!/usr/bin/env python3
"""
🥊 Qwen-powered Street Fighter 2 Agent
Works with existing wrapper.py without modifications
"""

import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import Dict, List, Tuple, Optional


class QwenStreetFighterAgent:
    """
    Qwen-powered agent for Street Fighter 2
    Uses existing wrapper.py environment without modifications
    """

    def __init__(self, model_path: str = "Qwen/Qwen3-4B-Instruct-2507"):
        """
        Initialize the Qwen agent
        
        Args:
            model_path: Path to Qwen model (local or HuggingFace)
        """
        # Initialize Qwen model
        print(f"🤖 Loading Qwen model from: {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Download tokenizer first
        print("📥 Step 1/2: Downloading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Download model second (sequential download)
        print("📥 Step 2/2: Downloading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,  # Updated parameter name
            device_map="auto",
            trust_remote_code=True,
            resume_download=True,  # Resume if interrupted
            max_memory={0: "6GB"} if torch.cuda.is_available() else None  # Limit memory usage
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
        Extract structured features from game state info
        
        Args:
            info: Game state info dictionary from environment
            
        Returns:
            Dictionary of structured game features
        """
        features = {
            # Player status
            "agent_hp": info.get("agent_hp", 176),
            "agent_x": info.get("agent_x", 0),
            "agent_y": info.get("agent_y", 0),
            "agent_status": info.get("agent_status", 0),
            "agent_victories": info.get("agent_victories", 0),
            
            # Enemy status  
            "enemy_hp": info.get("enemy_hp", 176),
            "enemy_x": info.get("enemy_x", 0),
            "enemy_y": info.get("enemy_y", 0),
            "enemy_status": info.get("enemy_status", 0),
            "enemy_victories": info.get("enemy_victories", 0),
            "enemy_character": info.get("enemy_character", 0),
            
            # Game status
            "score": info.get("score", 0),
            "round_countdown": info.get("round_countdown", 99),
            "reset_countdown": info.get("reset_countdown", 0),
        }
        
        # Calculate derived features
        features["hp_advantage"] = features["agent_hp"] - features["enemy_hp"]
        features["distance"] = abs(features["agent_x"] - features["enemy_x"])
        features["height_diff"] = features["agent_y"] - features["enemy_y"]
        features["agent_hp_percent"] = (features["agent_hp"] / 176.0) * 100
        features["enemy_hp_percent"] = (features["enemy_hp"] / 176.0) * 100
        
        # Determine relative position
        if features["agent_x"] < features["enemy_x"]:
            features["facing"] = "right"  # enemy is to the right
        else:
            features["facing"] = "left"   # enemy is to the left
            
        return features
    
    def create_reasoning_prompt(self, features: Dict, recent_actions: List[str]) -> str:
        """
        Create a natural language prompt for Qwen to reason about the game state
        
        Args:
            features: Extracted game features
            recent_actions: List of recent actions taken
            
        Returns:
            Formatted prompt string
        """
        # Character mapping
        character_names = {0: "Unknown", 1: "Bison", 2: "Other"}
        enemy_name = character_names.get(features["enemy_character"], "Enemy")
        
        # Status interpretation
        def interpret_status(status_code):
            if status_code == 0:
                return "neutral"
            elif status_code < 10:
                return "attacking" 
            elif status_code < 20:
                return "blocking"
            elif status_code < 30:
                return "stunned"
            else:
                return "special_move"
        
        agent_state = interpret_status(features["agent_status"])
        enemy_state = interpret_status(features["enemy_status"])
        
        # Recent actions context
        recent_actions_str = ", ".join(recent_actions[-3:]) if recent_actions else "None"
        
        # Distance assessment
        if features["distance"] < 80:
            distance_desc = "Very Close - melee range"
        elif features["distance"] < 150:
            distance_desc = "Close - combo range"
        elif features["distance"] < 250:
            distance_desc = "Medium - projectile range"
        else:
            distance_desc = "Far - need to move closer"

        prompt = f"""You are an expert Street Fighter 2 player controlling Ken. Analyze the current game situation and decide the best action.

CURRENT GAME STATE:
===================
Ken (You):
- Health: {features['agent_hp']}/176 ({features['agent_hp_percent']:.1f}%)
- Position: ({features['agent_x']}, {features['agent_y']})
- Status: {agent_state}
- Rounds Won: {features['agent_victories']}
- Facing: {features['facing']}

Enemy ({enemy_name}):
- Health: {features['enemy_hp']}/176 ({features['enemy_hp_percent']:.1f}%)
- Position: ({features['enemy_x']}, {features['enemy_y']})
- Status: {enemy_state}
- Rounds Won: {features['enemy_victories']}

TACTICAL SITUATION:
===================
- HP Advantage: {features['hp_advantage']:+d} ({"Winning" if features['hp_advantage'] > 0 else "Losing" if features['hp_advantage'] < 0 else "Even"})
- Distance: {features['distance']} pixels ({distance_desc})
- Height Difference: {features['height_diff']:+d} pixels
- Time Remaining: {features['round_countdown']}
- Recent Actions: {recent_actions_str}

STRATEGY GUIDELINES:
====================
1. CLOSE RANGE (distance < 80): Use punches, kicks, throws
2. MEDIUM RANGE (80-250): Hadoken projectiles, move in for combos  
3. FAR RANGE (>250): Move closer, use projectiles
4. LOW HP: Play defensively, look for counter-attacks
5. HIGH HP ADVANTAGE: Aggressive pressure, corner enemy
6. ENEMY STUNNED: Go for big damage combos

AVAILABLE ACTIONS (choose number 0-{self.num_actions-1}):
{"".join([f"{i}: {action}" + ("" if i % 3 != 2 else "") for i, action in enumerate(self.action_meanings)])}

Think strategically about the best move considering:
- Your current position and health
- Enemy's state and vulnerabilities  
- Distance and timing for attacks
- Special moves vs basic attacks

Respond with your reasoning and chosen action number.

RESPONSE FORMAT:
Action: [number 0-{self.num_actions-1}]
Reasoning: [brief explanation]"""

        return prompt
    
    def query_qwen(self, prompt: str) -> str:
        """
        Query Qwen model with the reasoning prompt
        
        Args:
            prompt: Formatted prompt for reasoning
            
        Returns:
            Model's response containing reasoning and action
        """
        # Tokenize input
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
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
    
    def get_action(self, info: Dict, verbose: bool = False) -> Tuple[int, str]:
        """
        Get action decision from Qwen based on current game state
        
        Args:
            info: Game state info from environment
            verbose: Whether to print reasoning
            
        Returns:
            Tuple of (action_number, reasoning_text)
        """
        # Extract features
        features = self.extract_game_features(info)
        
        # Create prompt
        prompt = self.create_reasoning_prompt(features, self.action_history[-5:])
        
        # Query Qwen
        response = self.query_qwen(prompt)
        
        # Parse action
        action = self.parse_action_from_response(response)
        
        # Update action history
        action_name = self.action_meanings[action]
        self.action_history.append(action_name)
        
        # Keep history manageable
        if len(self.action_history) > 20:
            self.action_history = self.action_history[-20:]
        
        if verbose:
            print(f"\n🧠 Qwen Decision:")
            print(f"Distance: {features['distance']}, HP: {features['agent_hp']}/{features['enemy_hp']}")
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
        
        # Test action selection
        action, reasoning = agent.get_action(mock_info, verbose=True)
        
        print(f"\n✅ Test completed! Chosen action: {action}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Note: You'll need to download Qwen weights separately")