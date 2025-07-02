![win_rate](https://github.com/user-attachments/assets/266fb8bb-f97f-4572-a313-b6bcdc16fc5c)


win rate at 53.4%!!!!!!!!!!!!!!


python train.py --resume trained_models/ppo_sf2_7999760_steps.zip --total-timesteps 10000000 --learning-rate 1e-3 --render


python train.py   --total-timesteps 20000000   --num-envs 1   --learning-rate 3e-4 --render






python train.py --total-timesteps 6000000 --resume trained_models_cuda_discrete/ppo_sf2_cuda_discrete_4000000_steps.zip --learning-rate 2.0e-4


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





+------------------------------------+
| Step 1: `momentum_tracker.update()`|
+------------------------------------+
       |
       v
+------------------------------------+
| A single 18-dim feature vector     |  e.g., [0.5, -1.0, ...]
+------------------------------------+
       |
       v
+------------------------------------+
| Step 2: Append to `history` deque  |  (A list of the last 8 vectors)
+------------------------------------+
       |
       v
+------------------------------------+
| Step 3: Stack & Concatenate        |
|  - CNN features   [8, 512]         |
|  - OpenCV features [8, 2]          |
|  - Momentum feats [8, 18] <---     |
+------------------------------------+
       |
       v
+------------------------------------+
| A combined sequence `[8, 532]`     |
+------------------------------------+
       |
       v
+------------------------------------+
| Step 4: Convert to Tensor `[1, 8, 532]` |
+------------------------------------+
       |
       v
+------------------------------------+
|  `EnhancedVisionTransformer`       |  (Final input)
+------------------------------------+




"""
wrapper.py - Enhanced Vision Pipeline for Street Fighter II with Position & Score Tracking
Raw Frames → OpenCV → CNN ↘
                           Vision Transformer → Predictions
Health/Score/Position Data → Enhanced Momentum Tracker ↗
"""
