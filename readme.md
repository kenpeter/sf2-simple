__win rate at 63.2%__

<img width="516" height="570" alt="win" src="https://github.com/user-attachments/assets/59fe8e24-bafc-40d7-b628-7c8bd63aba12" />


[training_small.webm](https://github.com/user-attachments/assets/f2f6f763-b9a5-4aeb-8150-f2e86ebe5a55)




     ================================================================================
     … +82 lines (ctrl+r to expand)
  ⎿  Found 1 new diagnostic issue in 1 file (ctrl-r to expand)

● Perfect! 🎯 Now you have all the direct download URLs. Here's your manual download strategy:

  🚀 Manual Download (One file at a time):

  Step 1: Create directory

  mkdir -p ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/main/
  cd ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/main/

  Step 2: Download small config files (fast)

  wget 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/.gitattributes'
  wget 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/LICENSE'
  wget 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/README.md'
  wget 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/config.json'
  wget 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/generation_config.json'
  wget 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/merges.txt'
  wget 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/model.safetensors.index.json'
  wget 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/tokenizer.json'
  wget 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/tokenizer_config.json'
  wget 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/vocab.json'
  # ... and the rest of the config files

  Step 3: Download BIG files ONE AT A TIME

  # Download first model file (3.96GB)
  wget -c 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/model-00001-of-00003.safetensors'

  # Then second model file (3.99GB)  
  wget -c 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/model-00002-of-00003.safetensors'

  # Finally third model file (99.6MB)
  wget -c 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/model-00003-of-00003.safetensors'

  Step 4: Test

  python demo_qwen.py --episodes 3

  💡 Benefits:

  - ✅ One file at a time - no parallel downloads
  - ✅ Resume support - wget -c to resume if interrupted
  - ✅ Full bandwidth to single file
  - ✅ Clear progress - you see exactly what's downloading

  This way you get complete control over the download process!


python train.py --mode train --resume train/best_model_1420000.zip --total_timesteps 6000000

  ta.json 
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




┌─────────────────────────────────────────────────────────────────────────────────┐
│                            GAME ENVIRONMENT                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Raw Frame: (H, W, 3) RGB                                                     │
│  Game State: health, position, score, status                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          PREPROCESSING STAGE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Frame Resize: (128, 180, 3)                                                  │
│  Frame Stack: 8 frames → (24, 128, 180) for CNN input                        │
│  Strategic Processing: Extract 21 strategic features                           │
│  Button History: Track 12 previous button states                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      FEATURE EXTRACTION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐           │
│  │  CNN EXTRACTOR  │    │ STRATEGIC TRACK │    │ BUTTON HISTORY  │           │
│  │                 │    │                 │    │                 │           │
│  │ Input:          │    │ Input:          │    │ Input:          │           │
│  │ (24,128,180)    │    │ Game state info │    │ Previous action │           │
│  │                 │    │                 │    │                 │           │
│  │ Output:         │    │ Output:         │    │ Output:         │           │
│  │ 512-dim vector  │    │ 21-dim vector   │    │ 12-dim vector   │           │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘           │
│                                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      TEMPORAL FEATURE MANAGEMENT                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Visual History:    [8 × 512] = (8, 512)                                     │
│  Strategic History: [8 × 21]  = (8, 21)                                      │
│  Button History:    [8 × 12]  = (8, 12)                                      │
│                                                                                │
│  Combined Sequence: (8, 545) = (8, 512+21+12)                                │
│                                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   CROSS-ATTENTION VISION TRANSFORMER                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                    FEATURE GROUP PROCESSORS                             │  │
│  │                                                                         │  │
│  │  Visual:    (8, 512) → Visual Processor    → (8, 256)                 │  │
│  │  Strategy:  (8, 21)  → Strategy Processor  → (8, 256)                 │  │
│  │  Button:    (8, 12)  → Button Processor    → (8, 256)                 │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                    CROSS-ATTENTION LAYERS                               │  │
│  │                                                                         │  │
│  │  Learnable Query: "What button should I press now?" → (1, 256)        │  │
│  │                                                                         │  │
│  │  Visual Cross-Attention:    Q=(1,256), K=V=(8,256) → (1,256)         │  │
│  │  Strategy Cross-Attention:  Q=(1,256), K=V=(8,256) → (1,256)         │  │
│  │  Button Cross-Attention:    Q=(1,256), K=V=(8,256) → (1,256)         │  │
│  │                                                                         │  │
│  │  Multi-Head Attention: 8 heads per cross-attention layer              │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                           │
│                                    ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                      FEATURE FUSION                                     │  │
│  │                                                                         │  │
│  │  Concatenate: (1,256) + (1,256) + (1,256) = (1,768)                  │  │
│  │  Fusion Network: (1,768) → (1,256)                                    │  │
│  │                                                                         │  │
│  │  Temporal Attention: Query=(1,256), Key=Value=(8,768) → (1,256)       │  │
│  │  Final Features: (1,256) → squeeze → (256)                            │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PPO POLICY NETWORK                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐   │
│  │        POLICY BRANCH            │    │        VALUE BRANCH             │   │
│  │                                 │    │                                 │   │
│  │  Input: (256) processed features│    │  Input: (256) processed features│   │
│  │         ↓                       │    │         ↓                       │   │
│  │  FC Layer: 256 → 512            │    │  FC Layer: 256 → 512            │   │
│  │  ReLU + Dropout                 │    │  ReLU + Dropout                 │   │
│  │         ↓                       │    │         ↓                       │   │
│  │  FC Layer: 512 → 256            │    │  FC Layer: 512 → 256            │   │
│  │  ReLU + Dropout                 │    │  ReLU + Dropout                 │   │
│  │         ↓                       │    │         ↓                       │   │
│  │  Output: 57 action logits       │    │  Output: 1 value estimate       │   │
│  │  (Discrete action distribution) │    │  (State value V(s))             │   │
│  └─────────────────────────────────┘    └─────────────────────────────────┘   │
│                                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ACTION SELECTION & EXECUTION                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Action Sampling: Sample from 57-way categorical distribution                  │
│                   ↓                                                            │
│  Discrete Action: Integer index (0-56)                                        │
│                   ↓                                                            │
│  Action Converter: Map discrete index to button combination                    │
│                   ↓                                                            │
│  Multi-Binary: 12-dimensional binary vector                                   │
│                [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R]       │
│                   ↓                                                            │
│  Game Input: Execute button combination in game                               │
│                                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING FEEDBACK LOOP                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Reward Signal: Combat performance + combo bonuses                            │
│  PPO Loss: Policy loss + Value loss + Entropy bonus                          │
│  Gradient Update: Backprop through entire network                             │
│                   ↓                                                            │
│  Parameter Update: CNN → Cross-Attention → Policy/Value Networks              │
│                                                                                │
