
<img width="1219" height="914" alt="win" src="https://github.com/user-attachments/assets/d6f981f5-9386-4ee4-be90-8682f5acca9e" />



__win rate at 55%!!!!!!!!!!!!!!__




# cli 
python train.py \
   --max-episodes 200 \
   --learning-rate 3e-4 \
   --batch-size 28 \
   --features-dim 512 \
   --thinking-steps 6 \
   --buffer-capacity 30000 \
   --target-win-rate 0.6 \
   --eval-frequency 12 \
   --save-frequency 30 \
   --render
   --resume checkpoints/enhanced_rgb_checkpoint_ep_99.pth




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



