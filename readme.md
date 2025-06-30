




python train.py --resume trained_models/ppo_sf2_7999760_steps.zip --total-timesteps 10000000 --learning-rate 1e-3 --render




python train.py \
  --resume trained_models/ppo_sf2_7999760_steps.zip \
  --total-timesteps 50000000 \
  --learning-rate 0.0004 \
  --ent-coef 0.03 \
  --render



python logger.py --frames 5000 --state ken_bison_12.state --output clean_test.csv --render



{
  "info": {
    "continuetimer": {
      "address": 16744917,
      "type": "|u1"
    },
    "enemy_health": {
      "address": 16745154,
      "type": ">i2"
    },
    "enemy_matches_won": {
      "address": 16745559,
      "type": ">u4"
    },
    "health": {
      "address": 16744514,
      "type": ">i2"
    },
    "matches_won": {
      "address": 16744922,
      "type": "|u1"
    },
    "score": {
      "address": 16744936,
      "type": ">d4"
    }
  }
}
