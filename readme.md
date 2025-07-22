![win_rate](https://github.com/user-attachments/assets/266fb8bb-f97f-4572-a313-b6bcdc16fc5c)


__win rate at 53.4%!!!!!!!!!!!!!!__




python train.py --max-episodes 2000 --temperature 0.8 --visual-lr 2e-4 --render




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
