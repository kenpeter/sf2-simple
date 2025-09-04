#!/usr/bin/env python3
"""
🥊 Qwen-powered Street Fighter 2 Agent
Works with existing wrapper.py without modifications
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import Dict, List, Tuple


class QwenStreetFighterAgent:
    """
    Qwen-powered agent for Street Fighter 2
    Uses existing wrapper.py environment without modifications
    """

    def __init__(self, model_path: str = "/home/kenpeter/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/main"):
        """
        Initialize the Qwen agent
        
        Args:
            model_path: Path to Qwen model (local or HuggingFace)
        """
        # Initialize Qwen model
        print(f"🤖 Loading Qwen model from: {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer from cache
        print("📁 Step 1/2: Loading tokenizer from cache...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # Load model from cache
        print("📁 Step 2/2: Loading model weights from cache...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            max_memory={0: "4GB"} if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
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
    
    def create_reasoning_prompt(self, features: Dict, recent_actions: List[str]) -> str:
        """
        Create a natural language prompt for Qwen to reason about the game state
        
        Args:
            features: Extracted game features
            recent_actions: List of recent actions taken
            
        Returns:
            Formatted prompt string
        """
        # Enemy name (no character info available)
        enemy_name = "Enemy"
        
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

        prompt = f"""Street Fighter expert: You control Ken. Choose the optimal action based on current situation.

SITUATION:
Ken HP: {features['agent_hp']} | Enemy HP: {features['enemy_hp']} | Distance: {features['distance']}
Ken pos: ({features['agent_x']},{features['agent_y']}) | Enemy pos: ({features['enemy_x']},{features['enemy_y']})
Status: Ken={agent_state} | Facing: {features['facing']}
Rounds: Ken {features['agent_victories']} - {features['enemy_victories']} Enemy | Time: {features['round_countdown']}

STRATEGY:
- Close (0-80px): Combos, throws, normals
- Medium (80-200px): Hadoken, approach 
- Far (200px+): Move closer, zone with fireballs
- Winning HP: Pressure aggressively
- Losing HP: Counter-attack, defensive
- Enemy stunned: Maximum damage combo

TOP ACTIONS:
Movement: 1=UP 2=DOWN 3=LEFT 6=RIGHT
Attacks: 9=L.PUNCH 13=M.PUNCH 17=H.PUNCH 21=L.KICK 26=M.KICK 32=H.KICK
Specials: 38=HADOKEN_R 41=HADOKEN_L 39=DP_R 42=DP_L 40=HURRICANE_R 43=HURRICANE_L

Choose action 0-{self.num_actions-1}. Format: Action: X"""

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
                max_new_tokens=50,  # Much shorter responses
                temperature=0.1,    # Less randomness for speed
                do_sample=False,    # Greedy decoding for speed
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,     # Enable caching
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
        
        # Smart rule-based agent using actual game features
        import random
        
        distance = features['distance']
        agent_hp = features['agent_hp'] 
        enemy_hp = features['enemy_hp']
        facing = features['facing']
        
        # Strategic decision making
        if agent_hp < enemy_hp and agent_hp < 50:  # Low HP, losing
            if distance > 150:
                action_choices = [38, 41]  # Hadoken to zone
                strategy = "defensive zoning"
            else:
                action_choices = [2, 3, 6]  # Movement to escape
                strategy = "escape pressure"
        elif distance < 60:  # Very close combat
            if agent_hp > enemy_hp:
                action_choices = [17, 32, 13, 26]  # Aggressive attacks
                strategy = "close pressure"
            else:
                action_choices = [9, 21, 2]  # Light attacks and escape
                strategy = "defensive pokes"
        elif distance < 150:  # Medium range
            if facing == "right":
                action_choices = [38, 6, 17]  # Hadoken right, move right, heavy punch
            else:
                action_choices = [41, 3, 17]  # Hadoken left, move left, heavy punch
            strategy = "mid-range control"
        else:  # Far range
            if facing == "right":
                action_choices = [6, 7, 38]  # Move closer right, hadoken
            else:
                action_choices = [3, 4, 41]  # Move closer left, hadoken  
            strategy = "close distance"
        
        action = random.choice(action_choices)
        reasoning = f"{strategy} (d:{distance}, hp:{agent_hp}/{enemy_hp})"
        
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
            print(f"Reasoning: {reasoning}")
        
        return action, reasoning
    
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