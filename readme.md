🏆 VICTORY! Win #41/112 (Rate: 36.6%)
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.32e+03     |
|    ep_rew_mean          | -41.6        |
| time/                   |              |
|    fps                  | 1164         |
|    iterations           | 306          |
|    time_elapsed         | 8608         |
|    total_timesteps      | 10027008     |
| train/                  |              |
|    approx_kl            | 5.538472e-05 |
|    clip_fraction        | 0.016        |
|    clip_range           | 0.0251       |
|    entropy_loss         | -8.3         |
|    explained_variance   | 0.284        |
|    learning_rate        | 2.64e-06     |
|    loss                 | 12.7         |
|    n_updates            | 1220         |
|    policy_gradient_loss | 0.000136     |
|    value_loss           | 25.5         |
------------------------------------------




2.64e-06  




python train.py --resume trained_models/ppo_sf2_22601232_steps.zip --total-timesteps 10000000 --learning-rate 3e-4 --render



python train.py --total-timesteps 5000000 --learning-rate 3e-4 --render