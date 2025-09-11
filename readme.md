# Qwen Street Fighter Agent

__win rate at 63.2%__

<img width="516" height="570" alt="win" src="https://github.com/user-attachments/assets/59fe8e24-bafc-40d7-b628-7c8bd63aba12" />


[training_small.webm](https://github.com/user-attachments/assets/f2f6f763-b9a5-4aeb-8150-f2e86ebe5a55)

## How the Qwen Agent Code Works - Flow Diagram

```
START
  │
  ├─── INITIALIZATION PHASE
  │     │
  │     ├─ Load Qwen2.5-VL vision model from cache
  │     ├─ Setup 44 total actions (but focus on 7 basic ones)
  │     ├─ Initialize cooldown timers and action history
  │     └─ Ready to play
  │
  ├─── MAIN GAME LOOP
  │     │
  │     ├─ Get game observation (screen frame + game state)
  │     │
  │     ├─── TIMING CONTROL SYSTEM
  │     │     │
  │     │     ├─ Check if frame_counter % 30 == 0 (every 0.5 seconds)
  │     │     ├─ Check if action_cooldown <= 0 (not in recovery)
  │     │     │
  │     │     ├─ IF YES: Make new decision
  │     │     └─ IF NO: Use cached action or NO_ACTION
  │     │
  │     ├─── DECISION MAKING PROCESS (when allowed)
  │     │     │
  │     │     ├─ capture_game_frame()
  │     │     │   ├─ Convert numpy observation to PIL Image  
  │     │     │   └─ Handle different image formats
  │     │     │
  │     │     ├─ extract_game_features()
  │     │     │   ├─ Parse HP, positions, status from game info
  │     │     │   ├─ Calculate distance, HP advantage, facing direction
  │     │     │   └─ Build strategic context
  │     │     │
  │     │     ├─ create_unified_prompt()
  │     │     │   ├─ Analyze current situation (health, distance, positioning)
  │     │     │   ├─ Add frame history context (HP changes, movement)
  │     │     │   ├─ Create strategy based on distance:
  │     │     │   │   ├─ Close (<40px): Punch/Kick/Block
  │     │     │   │   ├─ Medium (<80px): Move closer
  │     │     │   │   └─ Far (>80px): Move forward or jump
  │     │     │   └─ Generate action prompt focusing on basic moves only
  │     │     │
  │     │     ├─ query_qwen_vl()
  │     │     │   ├─ Format messages with image + text prompt
  │     │     │   ├─ Process inputs through Qwen2.5-VL model
  │     │     │   ├─ Generate response (max 50 tokens, greedy decoding)
  │     │     │   └─ Return AI's text response
  │     │     │
  │     │     └─ parse_action_from_response()
  │     │         │
  │     │         ├─ Extract numbers from AI response
  │     │         ├─ Prioritize basic actions (0,1,2,3,6,9,21)
  │     │         ├─ Convert complex actions to basic equivalents
  │     │         ├─ Try keyword matching if no numbers found
  │     │         └─ Use cycling fallback as last resort
  │     │
  │     ├─── ACTION PROCESSING
  │     │     │
  │     │     ├─ Anti-repeat system: prevent same action >2 times
  │     │     ├─ Set cooldown based on action recovery frames
  │     │     ├─ Cache action and reasoning for future frames  
  │     │     └─ Update action history
  │     │
  │     ├─── EXECUTE ACTION
  │     │     │
  │     │     ├─ Send action number (0-43) to game environment
  │     │     ├─ Environment processes action and updates game state
  │     │     └─ Get reward/penalty based on combat effectiveness
  │     │
  │     └─ COOLDOWN MANAGEMENT
  │           │
  │           ├─ Decrement action_cooldown counter each frame
  │           ├─ Track frames_since_last_action
  │           └─ Block new decisions until cooldown expires
  │
  └─── EPISODE END
        │
        ├─ Reset all counters and history
        ├─ Clear frame buffers and cached actions
        └─ Ready for new episode

BASIC ACTIONS FOCUSED ON:
├─ 0 = NO_ACTION (wait/rest)
├─ 1 = JUMP (up movement)
├─ 2 = CROUCH (down/low block)
├─ 3 = LEFT (move left/block away)
├─ 6 = RIGHT (move right/block toward)  
├─ 9 = PUNCH (basic attack)
└─ 21 = KICK (basic attack)

TIMING SYSTEM:
├─ New decisions: Every 30 frames (0.5 seconds)
├─ Action cooldowns: Vary by move (punch=11f, kick=12f, jump=15f)
├─ Recovery frames: Must wait before next action
└─ Real-time play: 60fps game, 2 decisions per second max
```



 # Data Collection:
  # Initial collection
  python lora_train.py --collect-data --episodes 1 --no-train

  # Append more data  
  python lora_train.py --collect-data --episodes 20 --append-data --no-train

  # Training:
  # Fresh training
  python lora_train.py --data-path ./data/sf2_training_data.json --epochs 1 --batch-size 1

  # With checkpoints (every 100 steps)
  python lora_train.py --data-path ./data/sf2_training_data.json --epochs 3 --save-steps 100

  # Resume training from checkpoint
  python lora_train.py --resume-from ./sf2_lora_model/checkpoint-6 --epochs 1 --batch-size 1

  # Inference:
  # Base model only
  python play.py --episodes 1 --no-render --quiet

  # With latest checkpoint (final model)
  python play.py --episodes 1 --model /home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-3B-Instruct-AWQ --lora ./sf2_lora_model

  # With specific checkpoint
  python play.py --episodes 1 --model /home/kenpeter/.cache/huggingface/hub/Qwen2.5-VL-3B-Instruct-AWQ --lora
  ./sf2_lora_model/checkpoint-6

  Now you can delete the sf2_lora_inference directory since it's no longer needed! 🗑️

  



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




  wget https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/resolve/main/config.json
  wget https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/resolve/main/generation_config.json
  wget https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/resolve/main/preprocessor_config.json
  wget https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/resolve/main/chat_template.json


  wget https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/resolve/main/added_tokens.json
  wget https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/resolve/main/merges.txt
  wget https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/resolve/main/tokenizer.json
  wget https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/resolve/main/tokenizer_config.json
  wget https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/resolve/main/vocab.json

  wget https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/resolve/main/model.safetensors






● wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/config.json
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/generation_config.json
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/model.safetensors.index.json
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/tokenizer.json
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/tokenizer_config.json
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/vocab.json
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/merges.txt
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/special_tokens_map.json
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/added_tokens.json
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/preprocessor_config.json
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/chat_template.json
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/model-00001-of-00011.safetensors
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/model-00002-of-00011.safetensors
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/model-00003-of-00011.safetensors
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/model-00004-of-00011.safetensors
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/model-00005-of-00011.safetensors
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/model-00006-of-00011.safetensors
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/model-00007-of-00011.safetensors
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/model-00008-of-00011.safetensors
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/model-00009-of-00011.safetensors
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/model-00010-of-00011.safetensors
  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ/resolve/main/model-00011-of-00011.safetensors




  # Create directory structure
  mkdir -p ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/main/
  cd ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/main/

  # Download all files
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/.gitattributes
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/README.md
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.json
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/config.json
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/generation_config.json
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/merges.txt
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/model-00001-of-00005.safetensors
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/model-00002-of-00005.safetensors
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/model-00003-of-00005.safetensors
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/model-00004-of-00005.safetensors
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/model-00005-of-00005.safetensors
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/model.safetensors.index.json
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/preprocessor_config.json
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/tokenizer.json
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/tokenizer_config.json
  wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/vocab.json






  wget -c https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00001-of-00002.safetensors \
       https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00002-of-00002.safetensors \
       https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model.safetensors.index.json \
       https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/config.json \
       https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/generation_config.json \
       https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/preprocessor_config.json \
       https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/tokenizer.json \
       https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/vocab.json \
       https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/tokenizer_config.json \
       https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/merges.txt \
       https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/chat_template.json \
       https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/README.md \
       https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/LICENSE



  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00001-of-00002.safetensors
  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00002-of-00002.safetensors
  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model.safetensors.index.json

  
  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/config.json
  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/generation_config.json
  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/preprocessor_config.json

  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/tokenizer.json
  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/vocab.json
  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/tokenizer_config.json
  https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/merges.txt


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
