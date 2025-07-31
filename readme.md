# cnn + Energy base thinking + Energy base transformer plays street fighter 2 with 55% win rate


<img width="1219" height="914" alt="win" src="https://github.com/user-attachments/assets/d6f981f5-9386-4ee4-be90-8682f5acca9e" />



__win rate at 55%!!!!!!!!!!!!!!__


python train.py --num_episodes 100 --batch_size 64 --learning_rate 1e-4 --features_dim 256 --thinking_steps 6 --thinking_lr 0.025 --buffer_capacity 30000 --gamma 0.99 --contrastive_margin 1.0 --contrastive_weight 0.5 --max_grad_norm 1.0 --train_frequency 1 --log_frequency 10 --save_frequency 30 --max_episode_steps 3000 --verify_health





# cli 
python train.py \
  --num_episodes 100 \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --thinking_steps 8 \
  --thinking_lr 0.025 \
  --contrastive_weight 0.4 \
  --log_frequency 5 \
  --save_frequency 100 \
  --verify_health



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



