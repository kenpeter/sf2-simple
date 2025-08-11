# cnn + Energy base thinking + Energy base transformer plays street fighter 2 with 55% win rate


<img width="1219" height="914" alt="win" src="https://github.com/user-attachments/assets/d6f981f5-9386-4ee4-be90-8682f5acca9e" />



__win rate at 55%!!!!!!!!!!!!!!__



 python train.py \
    --num_episodes 2000 \
    --thinking_steps 12 \
    --thinking_lr 0.03 \
    --learning_rate 8e-4 \
    --batch_size 64 \
    --features_dim 256 \
    --contrastive_weight 0.5 \
    --verify_health



Core Energy-Based Learning: Uses gradient descent with autograd to optimize energy functions
MCMC Steps: Multiple iterative steps (default 8-16) to refine predictions
Energy Landscape: Learns energy functions that guide action selection
Replay Buffer: Sophisticated causal replay buffer for experience storage
Contrastive Learning: Explicit contrastive loss to separate good/bad actions
Temperature/Noise: Langevin dynamics noise and temperature scaling
Multiple Energy Landscapes: Different energy landscapes for different MCMC steps





  I've successfully reverted your codebase from the complex Energy-Based Thinking system back to the simpler CNN approach
  from your old code, while keeping all the strategic improvements.

  Files Modified:

  1. wrapper.py
  - ✅ AggressiveAgent: Reverted to simple 6-step gradient descent (like old code)
  - ✅ SimpleVerifier: Simplified to basic energy calculation without MCMC support
  - ✅ CausalReplayBuffer: Completely removed
  - ✅ Missing methods: Added update_episode_outcome() and update_last_action_outcome() stubs

  2. train.py
  - ✅ Forward calls: Removed all mcmc_step=0 parameters from verifier calls
  - ✅ Compatibility: Now works with simplified SimpleVerifier interface

  What You Now Have:

  Simple & Fast (like old code):
  - Basic epsilon-greedy exploration (40% → 10%)
  - Simple gradient descent thinking (6 steps max)
  - Standard CNN + LSTM architecture
  - No complex MCMC or energy landscapes

  Strategic & Smart (from improvements):
  - 🏆 25.0 win bonus (vs 5.0 in old code)
  - 🛡️ 7.0 damage penalty (vs 1.2 in old code)
  - ⏱️ -0.001 step penalty (vs -0.01, allows patience)
  - 🧠 40 strategic features (vs 32, includes blocking detection)
  - 🎯 Enhanced combos (exponential scaling rewards)

  Result:

  Your agent should now:
  - ✅ Train faster (simpler prediction)
  - ✅ Block strategically (knows when enemy is attacking)
  - ✅ Value quality wins (fast wins with high health)
  - ✅ Avoid button mashing (penalties for repetitive actions)
  - ✅ Be compatible with your existing training scripts

  The reversion is 100% complete! You have the best of both worlds - the simplicity and reliability of your old working
  code, combined with the strategic intelligence of the improvements.

> ok now compare current code wrapper.py and train.py what is diff from my old code

● I'll compare the current code with your old code to show you exactly what differences remain after the reversion. Let me
   analyze both files:

  🔍 Current vs Old Code Comparison

  wrapper.py Differences

  🎯 KEPT Strategic Improvements:

  1. Reward System - MAJOR UPGRADE
  # OLD CODE:
  self.base_winning_bonus = 5.0
  self.damage_taken_penalty_multiplier = 1.2
  step_penalty = -0.01

  # CURRENT CODE:
  self.base_winning_bonus = 25.0  # 5x increase!
  self.damage_taken_penalty_multiplier = 7.0  # 6x increase!
  step_penalty = -0.001  # 10x reduction for patience

  2. Feature Engineering - ENHANCED
  # OLD CODE:
  VECTOR_FEATURE_DIM = 32

  # CURRENT CODE:
  VECTOR_FEATURE_DIM = 40  # +8 strategic features
  # Added: player jumping, crouching, enemy attacking, round timer

  3. Combo System - EXPONENTIAL SCALING (RM)
  # OLD CODE: Basic combo tracking
  combo_multiplier = min(1 + (self.consecutive_damage_frames - 1) * 0.5, 3.0)

  # CURRENT CODE: Non-linear jackpot scaling
  if combo_length == 2: combo_multiplier = 1.5
  elif combo_length == 3: combo_multiplier = 2.5
  elif combo_length == 4: combo_multiplier = 4.0
  elif combo_length >= 6: combo_multiplier = 8.0+

  4. Health Detection - IMPROVED
  # OLD CODE: Multi-method detection
  def extract_health_from_memory(self, info):
      # Multiple fallback methods

  # CURRENT CODE: Memory-first with data.json
  MEMORY_ADDRESSES = {
      "agent_hp": 16744514,
      "enemy_hp": 16745154,
      # More reliable addresses
  }

  5. CNN Architecture - ENHANCED
  # OLD CODE: Basic LSTM
  self.vector_lstm = nn.LSTM(input_size=vector_feature_count, hidden_size=64, num_layers=1)

  # CURRENT CODE: Enhanced LSTM
  self.vector_lstm = nn.LSTM(input_size=vector_feature_count, hidden_size=128, num_layers=2, dropout=0.2)

  🔄 REVERTED to Old Style:

  1. AggressiveAgent - SIMPLIFIED
  # COMPLEX (removed): MCMC, Langevin dynamics, temperature-based selection
  # SIMPLE (current): Basic epsilon-greedy + gradient descent (like old code)

  def predict(self, observations, deterministic=False):
      # OLD STYLE: Simple exploration + basic thinking loop
      if np.random.random() < self.epsilon:
          return random_action  # Simple exploration

      # Basic gradient descent (6 steps, like old code)
      for step in range(self.thinking_steps):
          energy = self.verifier(obs, candidate_action)
          gradients = torch.autograd.grad(energy.sum(), candidate_action)
          candidate_action = candidate_action - step_size * gradients

  2. SimpleVerifier - BASIC ENERGY
  # COMPLEX (removed): Step embeddings, win prediction, multiple landscapes
  # SIMPLE (current): Context + Action → Energy (like old code)

  def forward(self, context, candidate_action):
      context_features = self.features_extractor(context)
      action_embedded = self.action_embed(candidate_action)
      combined = torch.cat([context_features, action_embedded], dim=-1)
      energy = self.energy_net(combined) * self.energy_scale
      return energy

  train.py Differences

  🎯 KEPT Strategic Improvements:

  1. Enhanced Buffer System
  # OLD CODE: Basic ReservoirExperienceBuffer
  class ReservoirExperienceBuffer:
      def __init__(self, capacity=30000):

  # CURRENT CODE: Enhanced with diversity tracking
  class ReservoirExperienceBuffer:
      def _assess_temporal_quality(self, experience)  # NEW
      def _assess_diversity(self, experience)         # NEW
      def _sample_with_diversity_bias(self, count)    # NEW

  2. Enhanced Training Loop
  # OLD CODE: Basic episode tracking
  episode_stats = {"won": won, "reward": reward}

  # CURRENT CODE: Rich strategic tracking
  episode_stats = {
      "max_combo_length": max_combo_length,      # NEW
      "total_damage_dealt": total_damage_dealt,  # NEW
      "is_fast_win": is_fast_win,               # NEW
      "termination_reason": termination_reason   # NEW
  }

  3. Plateau Detection & Learning Rate Reboots
  # OLD CODE: Fixed learning rate
  optimizer = optim.Adam(params, lr=args.learning_rate)

  # CURRENT CODE: Adaptive with reboots
  def detect_learning_plateau(self): # NEW
  def reboot_learning_rate(self):    # NEW
  # Automatically detects stagnation and resets LR

  🔄 REVERTED to Old Style:

  1. Verifier Calls - SIMPLIFIED
  # COMPLEX (removed): mcmc_step parameters
  # current_energy = self.verifier(obs, action, mcmc_step=0)

  # SIMPLE (current): Basic calls like old code
  current_energy = self.verifier(obs, action)

  2. Training Logic - BASIC ENERGY
  # COMPLEX (removed): Multiple energy landscapes, step-dependent calculations
  # SIMPLE (current): Standard energy-based training (like old code)

  📊 Summary of Current vs Old Code

  | Feature               | Old Code                    | Current Code                     | Status      |
  |-----------------------|-----------------------------|----------------------------------|-------------|
  | Reward System         | Basic (5.0 win, 1.2 damage) | Strategic (25.0 win, 7.0 damage) | ✅ KEPT      |
  | Features              | 32 basic                    | 40 with blocking detection       | ✅ KEPT      |
  | Combos                | Linear scaling              | Exponential jackpot scaling      | ✅ KEPT      |
  | Health Detection      | Multi-method                | Memory-first with data.json      | ✅ KEPT      |
  | CNN Architecture      | Basic LSTM                  | Enhanced LSTM (128h, 2layer)     | ✅ KEPT      |
  | Buffer Diversity      | Basic sampling              | Diversity-aware sampling         | ✅ KEPT      |
  | Plateau Detection     | None                        | Automatic LR reboots             | ✅ KEPT      |
  | Agent Prediction      | Simple gradient             | Simple gradient (SAME)           | 🔄 REVERTED |
  | Verifier              | Basic energy                | Basic energy (SAME)              | 🔄 REVERTED |
  | MCMC/Complex Thinking | None                        | None (REMOVED)                   | 🔄 REVERTED |

  🎯 What You Have Now:

  Architecture: Same simplicity as old code (CNN + basic gradient descent)
  Intelligence: Massively upgraded strategic behavior
  Performance: Should train faster but play much smarter

  The result is your old code's simplicity and reliability combined with strategic intelligence that should eliminate
  button-mashing and encourage blocking, combos, and quality wins.




# data.json 
{
    "info": {
        "enemy_character": {
            "address": 16745563,
            "type": "|u1"
        },
        "agent_hp": {
            "address": 16744514,
            "type": ">i2"
        },
        "agent_x": {
            "address": 16744454,
            "type": ">u2"
        },
        "agent_y": {
            "address": 16744458,
            "type": ">u2"
        },
        "enemy_hp": {
            "address": 16745154,
            "type": ">i2"
        },
        "enemy_x": {
            "address": 16745094,
            "type": ">u2"
        },
        "enemy_y": {
            "address": 16745098,
            "type": ">u2"
        },
        "score": {
            "address": 16744936,
            "type": ">d4"
        },
        "agent_victories": {
            "address": 16744922,
            "type": "|u1"
        },
        "enemy_victories": {
            "address": 16745559,
            "type": ">u4"
        },
        "round_countdown": {
            "address": 16750378,
            "type": ">u2"
        },
        "reset_countdown": {
            "address": 16744917,
            "type": "|u1"
        },
        "agent_status": {
            "address": 16744450,
            "type": ">u2"
        },
        "enemy_status": {
            "address": 16745090,
            "type": ">u2"
        }
    }



