#!/usr/bin/env python3  # Shebang line to run script with python3 directly
"""
🥊 Qwen-powered Street Fighter 2 Agent
Works with existing wrapper.py without modifications
"""

# Import PyTorch for deep learning functionality
import torch  
# Import HuggingFace transformers for vision models
from transformers import AutoModelForVision2Seq, AutoProcessor
# Import NumPy for numerical array operations
import numpy as np  
from PIL import Image  # Import PIL for image processing
import re  # Import regular expressions for text pattern matching
from typing import Dict, Tuple  # Import typing hints for better code documentation


class QwenStreetFighterAgent:  # Define main agent class for Street Fighter 2 AI
    """
    Qwen-powered agent for Street Fighter 2
    Uses existing wrapper.py environment without modifications
    """

    def __init__(self, model_path: str = "/home/kenpeter/.cache/huggingface/hub/SmolVLM-Instruct", force_cpu: bool = False):  # Constructor method for agent initialization
        """
        Initialize the Qwen agent
        
        Args:
            model_path: Path to Qwen model (local or HuggingFace)
            force_cpu: Force CPU usage even if CUDA is available
        """
        # Initialize Qwen model
        print(f"🤖 Loading Qwen model from: {model_path}")  # Print model loading status
        self.device = "cpu" if (force_cpu or not torch.cuda.is_available()) else "cuda"  # Set device preference
        
        # Load processor and model for vision
        print("📁 Step 1/2: Loading processor from cache...")  # Print loading status for processor
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)  # Load tokenizer and image processor
        
        # Load vision model from cache
        print("📁 Step 2/2: Loading Qwen2.5-VL model from cache...")  # Print loading status for model
        self.model = AutoModelForVision2Seq.from_pretrained(  # Load the vision-language model
            model_path,  # Model path
            dtype=torch.float16,  # Use 16-bit precision for memory efficiency
            device_map="cpu" if not torch.cuda.is_available() else "auto",  # Use CPU if CUDA unavailable
            trust_remote_code=True,  # Allow custom code in model
            local_files_only=True,  # Only use local cached files
            max_memory={0: "3GB"} if torch.cuda.is_available() else None,  # Reduce GPU memory usage to 3GB
            torch_dtype=torch.float16  # Explicitly set torch dtype
        )
        print(f"✅ Qwen model loaded successfully on {self.device}")  # Print successful loading message
        
        # Action space - matches discretizer.py
        self.action_meanings = [  # Define all possible actions the agent can take
            "NO_ACTION",           # 0 - Do nothing action
            "UP",                  # 1 - Jump upward
            "DOWN",                # 2 - Crouch downward
            "LEFT",                # 3 - Move left
            "UP_LEFT",             # 4 - Jump diagonally left
            "DOWN_LEFT",           # 5 - Crouch walk left
            "RIGHT",               # 6 - Move right
            "UP_RIGHT",            # 7 - Jump diagonally right
            "DOWN_RIGHT",          # 8 - Crouch walk right
            "LIGHT_PUNCH",         # 9 - Quick punch attack
            "LIGHT_PUNCH_DOWN",    # 10 - Crouching light punch
            "LIGHT_PUNCH_LEFT",    # 11 - Light punch while moving left
            "LIGHT_PUNCH_RIGHT",   # 12 - Light punch while moving right
            "MEDIUM_PUNCH",        # 13 - Medium strength punch
            "MEDIUM_PUNCH_DOWN",   # 14 - Crouching medium punch
            "MEDIUM_PUNCH_LEFT",   # 15 - Medium punch while moving left
            "MEDIUM_PUNCH_RIGHT",  # 16 - Medium punch while moving right
            "HEAVY_PUNCH",         # 17 - Strong punch attack
            "HEAVY_PUNCH_DOWN",    # 18 - Crouching heavy punch
            "HEAVY_PUNCH_LEFT",    # 19 - Heavy punch while moving left
            "HEAVY_PUNCH_RIGHT",   # 20 - Heavy punch while moving right
            "LIGHT_KICK",          # 21 - Quick kick attack
            "LIGHT_KICK_DOWN",     # 22 - Crouching light kick
            "LIGHT_KICK_LEFT",     # 23 - Light kick while moving left
            "LIGHT_KICK_DOWN_LEFT", # 24 - Crouching light kick moving left
            "LIGHT_KICK_RIGHT",    # 25 - Light kick while moving right
            "MEDIUM_KICK",         # 26 - Medium strength kick
            "MEDIUM_KICK_DOWN",    # 27 - Crouching medium kick
            "MEDIUM_KICK_LEFT",    # 28 - Medium kick while moving left
            "MEDIUM_KICK_DOWN_LEFT", # 29 - Crouching medium kick moving left
            "MEDIUM_KICK_RIGHT",   # 30 - Medium kick while moving right
            "MEDIUM_KICK_DOWN_RIGHT", # 31 - Crouching medium kick moving right
            "HEAVY_KICK",          # 32 - Strong kick attack
            "HEAVY_KICK_DOWN",     # 33 - Crouching heavy kick
            "HEAVY_KICK_LEFT",     # 34 - Heavy kick while moving left
            "HEAVY_KICK_DOWN_LEFT", # 35 - Crouching heavy kick moving left
            "HEAVY_KICK_RIGHT",    # 36 - Heavy kick while moving right
            "HEAVY_KICK_DOWN_RIGHT", # 37 - Crouching heavy kick moving right
            "HADOKEN_RIGHT",       # 38 - Fireball special move facing right
            "DRAGON_PUNCH_RIGHT",  # 39 - Uppercut special move facing right
            "HURRICANE_KICK_RIGHT", # 40 - Spinning kick special move facing right
            "HADOKEN_LEFT",        # 41 - Fireball special move facing left
            "DRAGON_PUNCH_LEFT",   # 42 - Uppercut special move facing left
            "HURRICANE_KICK_LEFT", # 43 - Spinning kick special move facing left
        ]
        
        self.num_actions = len(self.action_meanings)  # Store total number of actions available
        
        # Action timing - frames each action takes to complete
        self.action_frames = {  # Dictionary mapping action IDs to animation frame durations
            0: 1,    # NO_ACTION - instant response
            1: 3,    # UP - short jump startup frames
            2: 3,    # DOWN - crouch animation frames
            3: 2,    # LEFT - walk animation frames
            6: 2,    # RIGHT - walk animation frames
            7: 5,    # UP_RIGHT - jump animation frames
            4: 5,    # UP_LEFT - jump animation frames
            5: 4,    # DOWN_LEFT - crouch walk animation frames
            8: 4,    # DOWN_RIGHT - crouch walk animation frames
            9: 8,    # LIGHT_PUNCH - fast attack animation frames
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
        
    def extract_game_features(self, info: Dict) -> Dict:  # Method to extract game features from info dict
        """
        Extract structured features from game state info based on ta.json schema
        
        Args:
            info: Game state info dictionary from environment
            
        Returns:
            Dictionary of structured game features
        """
        features = {  # Dictionary to store extracted game features
            # Player status (from ta.json)
            "agent_hp": info.get("agent_hp", 176),  # Get agent health points, default 176
            "agent_x": info.get("agent_x", 0),  # Get agent x-coordinate position
            "agent_y": info.get("agent_y", 0),  # Get agent y-coordinate position
            "agent_status": info.get("agent_status", 0),  # Get agent animation status
            "agent_victories": info.get("agent_victories", 0),  # Get agent wins count
            
            # Enemy status (from ta.json)
            "enemy_hp": info.get("enemy_hp", 176),  # Get enemy health points, default 176
            "enemy_x": info.get("enemy_x", 0),  # Get enemy x-coordinate position
            "enemy_y": info.get("enemy_y", 0),  # Get enemy y-coordinate position
            "enemy_victories": info.get("enemy_victories", 0),  # Get enemy wins count
            
            # Game status (from ta.json)
            "score": info.get("score", 0),  # Get current score
            "round_countdown": info.get("round_countdown", 99),  # Get time remaining in round
        }
        
        # Calculate derived features (only from available data)
        features["hp_advantage"] = features["agent_hp"] - features["enemy_hp"]  # Calculate health advantage
        features["distance"] = abs(features["agent_x"] - features["enemy_x"])  # Calculate horizontal distance
        features["height_diff"] = features["agent_y"] - features["enemy_y"]  # Calculate vertical distance
        
        # Determine relative position
        if features["agent_x"] < features["enemy_x"]:  # If agent is to the left of enemy
            features["facing"] = "right"  # Agent should face right
        else:  # If agent is to the right of enemy
            features["facing"] = "left"  # Agent should face left
            
        return features  # Return the features dictionary
    
    def capture_game_frame(self, observation) -> Image.Image:  # Method to convert observation to PIL Image
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
                dummy_frame = np.zeros((224, 320, 3), dtype=np.uint8)  # Create blank RGB frame
                return Image.fromarray(dummy_frame)  # Convert to PIL Image and return
            
            # Convert numpy array to PIL Image
            if observation.dtype != np.uint8:  # Check if values need scaling
                observation = (observation * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
            
            # Ensure proper shape for image
            if len(observation.shape) == 3 and observation.shape[2] in [3, 4]:  # Check for RGB/RGBA format
                # RGB or RGBA image
                if observation.shape[2] == 4:  # Check if has alpha channel
                    observation = observation[:, :, :3]  # Remove alpha channel, keep RGB only
                image = Image.fromarray(observation)  # Convert numpy array to PIL Image
            elif len(observation.shape) == 2:  # Check for grayscale format
                # Grayscale - convert to RGB
                image = Image.fromarray(observation).convert('RGB')  # Convert grayscale to RGB
            else:  # Handle unexpected formats
                # Unexpected format - create dummy image
                dummy_frame = np.zeros((224, 320, 3), dtype=np.uint8)  # Create blank RGB frame
                return Image.fromarray(dummy_frame)  # Convert to PIL Image and return
            
            return image  # Return the processed PIL Image
        else:  # If not numpy array
            # If already PIL Image, return as-is
            return observation  # Return observation unchanged
    
    def create_hybrid_prompt(self, features: Dict) -> str:  # Method to create prompt for vision model
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

Action:"""  # Create formatted prompt string with game state and action options

        return prompt  # Return the formatted prompt string
    
    def query_smolvlm(self, image: Image.Image, prompt: str) -> str:  # Method to query vision-language model
        """
        Query SmolVLM model with image and prompt
        
        Args:
            image: Game frame as PIL Image
            prompt: Text prompt for analysis
            
        Returns:
            Model's response containing reasoning and action
        """
        # Create messages for SmolVLM (Idefics3 format)
        messages = [  # Create message list in chat format
            {  # User message containing image and text
                "role": "user",  # Set role as user
                "content": [  # Content list with image and text
                    {"type": "image"},  # Image component
                    {"type": "text", "text": prompt}  # Text prompt component
                ]
            }
        ]
        
        # Create formatted text input
        text_input = self.processor.apply_chat_template(  # Apply chat template formatting
            messages, images=[image], add_generation_prompt=True  # Include generation prompt
        )
        
        # Process the text and image to get tensors
        inputs = self.processor(  # Process inputs for model
            text=text_input,  # Formatted text input
            images=[image],  # Image input
            return_tensors="pt"  # Return PyTorch tensors
        )
        
        # Safely move tensors to device with error handling
        try:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move tensors to correct device
        except RuntimeError as e:
            print(f"⚠️ Error moving tensors to device: {e}")
            # Fallback to CPU if CUDA fails
            self.device = "cpu"
            self.model = self.model.to("cpu")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        # Generate response with minimal tokens for speed
        with torch.no_grad():  # Disable gradient computation for inference
            try:
                outputs = self.model.generate(  # Generate response from model
                    **inputs,  # Pass all input tensors
                    max_new_tokens=8,   # Very short response for speed
                    do_sample=False,    # Use greedy decoding (deterministic)
                    pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,  # Set padding token with fallback
                    eos_token_id=self.processor.tokenizer.eos_token_id,  # Explicit EOS token
                )
            except RuntimeError as cuda_error:
                print(f"⚠️ CUDA error during generation: {cuda_error}")
                # Move everything to CPU and retry
                if self.device != "cpu":
                    print("🔄 Falling back to CPU inference...")
                    self.device = "cpu"
                    self.model = self.model.to("cpu")
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    
                    outputs = self.model.generate(  # Retry on CPU
                        **inputs,
                        max_new_tokens=8,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )
                else:
                    raise cuda_error  # Re-raise if already on CPU
        
        # Decode response  
        response = self.processor.decode(outputs[0], skip_special_tokens=True)  # Decode tokens to text
        # Extract just the action part after the prompt
        if "Action:" in response:  # Check if Action: marker exists
            response = response.split("Action:")[-1].strip()  # Extract text after Action: marker
        return response.strip()  # Return cleaned response
    
    def analyze_frame_context(self, image: Image.Image, features: Dict) -> str:  # Method to analyze visual frame context with OpenCV
        """
        Analyze the visual frame for additional context using OpenCV computer vision
        
        Args:
            image: Game frame as PIL Image
            features: Game state features
            
        Returns:
            Additional context from visual analysis
        """
        import numpy as np  # Import numpy for array operations
        import cv2  # Import OpenCV for computer vision
        
        # Convert PIL image to numpy for analysis
        frame = np.array(image)  # Convert PIL Image to numpy array
        
        # Analyze frame characteristics
        context = []  # Initialize empty list for context markers
        
        # BASIC ANALYSIS (keep existing for compatibility)
        brightness = np.mean(frame)  # Calculate average brightness across all pixels
        if brightness > 140:  # Check if frame is very bright
            context.append("bright_flash")  # Indicates special moves or hits
        elif brightness < 80:  # Check if frame is dark
            context.append("dark_frame")    # Indicates normal game state
        
        # ENHANCED OPENCV ANALYSIS
        if len(frame.shape) == 3:  # Check if frame has color channels
            # Convert to different color spaces for better analysis
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # Convert to HSV for better color detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale for edge detection
            
            # Edge detection for character and projectile detection
            edges = cv2.Canny(gray_frame, 50, 150)  # Detect edges using Canny edge detector
            edge_density = np.sum(edges > 0) / edges.size  # Calculate edge density
            
            if edge_density > 0.15:  # High edge density indicates action
                context.append("high_action")  # Lots of movement/attacks happening
            elif edge_density < 0.05:  # Low edge density indicates calm state
                context.append("calm_state")  # Little movement
            
            # Detect blue objects (fireballs) using HSV color space
            blue_mask = cv2.inRange(hsv_frame, np.array([100, 50, 50]), np.array([130, 255, 255]))  # Blue color range
            blue_pixels = np.sum(blue_mask > 0)  # Count blue pixels
            if blue_pixels > 500:  # Threshold for detecting significant blue objects
                context.append("blue_projectile")  # Likely fireball detected
            
            # Detect red/orange flashes (hits, special effects)
            red_mask = cv2.inRange(hsv_frame, np.array([0, 50, 50]), np.array([20, 255, 255]))  # Red color range
            orange_mask = cv2.inRange(hsv_frame, np.array([10, 50, 50]), np.array([30, 255, 255]))  # Orange color range
            hit_pixels = np.sum(red_mask > 0) + np.sum(orange_mask > 0)  # Count red/orange pixels
            if hit_pixels > 800:  # Threshold for detecting hits/explosions
                context.append("red_flash")  # Hit or explosion effect detected
            
            # Analyze motion using frame regions
            height, width = frame.shape[:2]  # Get frame dimensions
            left_region = gray_frame[:, :width//3]  # Left third of frame
            right_region = gray_frame[:, 2*width//3:]  # Right third of frame
            center_region = gray_frame[:, width//3:2*width//3]  # Center third of frame
            
            # Calculate activity levels using standard deviation
            left_activity = np.std(left_region)  # Left side activity
            right_activity = np.std(right_region)  # Right side activity
            center_activity = np.std(center_region)  # Center activity
            
            # Determine where most action is happening
            if center_activity > max(left_activity, right_activity) * 1.3:  # Center has most activity
                context.append("center_action")  # Action in center of screen
            elif left_activity > right_activity * 1.5:  # Left side more active
                context.append("left_active")   # More visual activity on left side
            elif right_activity > left_activity * 1.5:  # Right side more active
                context.append("right_active")  # More visual activity on right side
            
            # Detect character silhouettes using contour detection
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
            large_contours = [c for c in contours if cv2.contourArea(c) > 1000]  # Filter large contours (characters)
            
            if len(large_contours) >= 2:  # Two characters visible
                context.append("both_visible")  # Both fighters clearly visible
            elif len(large_contours) == 1:  # One character prominent
                context.append("one_prominent")  # One fighter more visible
        
        return context  # Return list of visual context indicators
    
    def enhanced_decision_making(self, features: Dict, image: Image.Image, prompt: str) -> Tuple[int, str]:  # Enhanced AI decision making with OpenCV + SmolVLM
        """
        Enhanced decision making combining OpenCV analysis with SmolVLM reasoning
        
        Args:
            features: Game state features
            image: Current game frame for visual analysis  
            prompt: Formatted prompt for SmolVLM
            
        Returns:
            Tuple of (action_number, reasoning_text)
        """
        # Get enhanced visual context using OpenCV
        visual_context = self.analyze_frame_context(image, features)  # Get visual cues from OpenCV analysis
        
        # Create enhanced prompt with visual context
        enhanced_prompt = prompt + f"\n\nVisual Context: {', '.join(visual_context)}\n\nBased on the image and context, choose the best action number:"  # Add visual context to prompt
        
        # Use SmolVLM for actual vision-language reasoning
        try:  # Try to get SmolVLM decision
            vlm_response = self.query_smolvlm(image, enhanced_prompt)  # Get SmolVLM decision
            action = self.parse_action_from_response(vlm_response)  # Parse action from response
            reasoning = f"SmolVLM: {vlm_response} | Context: {', '.join(visual_context)}"  # Combine reasoning
            
            # Validate action and fall back to rule-based if needed
            if 0 <= action < self.num_actions:  # Check if action is valid
                return action, reasoning  # Return SmolVLM decision
            else:  # If invalid action, fall back to rule-based
                return self.strategic_decision_making(features, image)  # Fall back to rule-based
                
        except Exception as e:  # If SmolVLM fails, fall back to rule-based
            print(f"⚠️ SmolVLM failed: {e}, falling back to rule-based")  # Print error message
            return self.strategic_decision_making(features, image)  # Fall back to rule-based decision
    
    def strategic_decision_making(self, features: Dict, image: Image.Image) -> Tuple[int, str]:  # Main AI decision making method
        """
        Intelligent rule-based decision making with proper reasoning and frame analysis
        
        Args:
            features: Game state features
            image: Current game frame for visual analysis
            
        Returns:
            Tuple of (action_number, reasoning_text)
        """
        distance = features['distance']  # Get distance between characters
        hp_advantage = features['hp_advantage']  # Get health point advantage
        facing = features['facing']  # Get which direction agent should face
        
        # Decrement action cooldown
        if self.action_cooldown > 0:  # Check if still in animation cooldown
            self.action_cooldown -= 1  # Decrease cooldown counter
            # Still in cooldown - return NO_ACTION or continue previous action
            if self.action_cooldown > 0:  # If still need to wait
                action = 0  # NO_ACTION - wait for animation to finish
                reasoning = f"waiting for {self.action_meanings[self.last_executed_action]} to complete ({self.action_cooldown} frames left)"  # Explain waiting
                return action, reasoning  # Return wait action and reason
        
        # Force action variety if stuck repeating same action
        force_movement = False  # Initialize movement forcing flag
        if self.action_repeat_count > 3:  # Check if repeating action too much
            force_movement = True  # Force different action to avoid repetition
            
        # Check if distance hasn't changed (stuck position)  
        distance_change = abs(distance - self.last_distance) if self.last_distance > 0 else 999  # Calculate position change
        if distance_change < 10 and self.frame_counter > 20:  # Check if stuck in same position
            force_movement = True  # Force movement to break out of stuck state
            
        self.last_distance = distance  # Store current distance for next frame
        
        # Analyze frame for visual context
        visual_context = self.analyze_frame_context(image, features)  # Get visual cues from frame analysis
        
        # Adjust strategy based on visual cues
        base_reasoning = ""  # Initialize reasoning string
        if "bright_flash" in visual_context:  # Check for bright flash indicator
            base_reasoning += "Flash detected - "  # Add flash detection to reasoning
        if "blue_projectile" in visual_context:  # Check for blue projectile indicator
            base_reasoning += "Projectile seen - "  # Add projectile detection to reasoning
        if "red_flash" in visual_context:  # Check for red flash indicator
            base_reasoning += "Hit detected - "  # Add hit detection to reasoning
        
        # Force movement if stuck
        if force_movement:  # Check if need to force movement
            import random  # Import random for action selection
            movement_actions = [3, 6, 1, 2]  # LEFT, RIGHT, UP, DOWN movement actions
            action = random.choice(movement_actions)  # Choose random movement action
            reasoning = base_reasoning + f"breaking pattern, forced movement (repeated {self.action_repeat_count}x)"  # Explain forced movement
            
        # React to visual cues first
        elif "blue_projectile" in visual_context and distance > 100:  # Check for distant projectile
            # Enemy projectile detected - dodge or counter
            action = 1  # UP (jump to avoid)
            reasoning = base_reasoning + "jumping to avoid projectile"  # Explain jump action
            
        elif "bright_flash" in visual_context and distance < 100:  # Check for close flash
            # Special move or hit flash detected - be defensive  
            if facing == "right":  # Check facing direction
                action = 42  # DRAGON_PUNCH_LEFT (away from enemy)
                reasoning = base_reasoning + "defensive dragon punch after flash"  # Explain defensive move
            else:  # If facing left
                action = 39  # DRAGON_PUNCH_RIGHT
                reasoning = base_reasoning + "defensive dragon punch after flash"  # Explain defensive move
                
        elif "red_flash" in visual_context:  # Check for red flash (hit indicator)
            # Hit detected - follow up or counter
            if distance < 80:  # Check if close range
                action = 17  # HEAVY_PUNCH
                reasoning = base_reasoning + "following up after hit with heavy attack"  # Explain follow-up attack
            else:  # If medium range
                action = 40 if facing == "right" else 43  # HURRICANE_KICK based on facing
                reasoning = base_reasoning + "hurricane kick to capitalize on hit"  # Explain hurricane kick
                
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
            
            # Enhanced decision making: OpenCV analysis + SmolVLM reasoning
            action, response = self.enhanced_decision_making(features, image, prompt)
            
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
        "enemy_status": 5  # Mock enemy animation status
    }
    
    try:  # Try to run test with error handling
        # Create agent (will download model if needed)
        agent = QwenStreetFighterAgent()  # Initialize agent with default model path
        
        # Test action selection with dummy frame
        dummy_frame = np.zeros((224, 320, 3), dtype=np.uint8)  # Create black RGB frame
        action, reasoning = agent.get_action(dummy_frame, mock_info, verbose=True)  # Test action selection
        
        print(f"\n✅ Test completed! Chosen action: {action}")  # Print successful test result
        
    except Exception as e:  # Catch any exceptions
        print(f"❌ Test failed: {e}")  # Print error message
        print("Note: You'll need to download Qwen weights separately")  # Print note about model requirements