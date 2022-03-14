# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from env import JetBotEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
import torch as th
import numpy as np

log_dir = "/home/omniverse02/.local/share/ov/pkg/isaac_sim-2021.2.1/standalone_examples/api/omni.isaac.jetbot/stable_baselines_example/mlp_policy"
# set headles to false to visualize training
my_env = JetBotEnv(headless=False)
check_env(my_env, warn=True)

policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[16, dict(pi=[64, 32], vf=[64, 32])])
policy = "MlpPolicy"
total_timesteps = 500000

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="jetbot_policy_checkpoint")

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record('random_value', value)
        return True

model = PPO(
    policy,
    my_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=10000,
    batch_size=1000,
    learning_rate=0.00025,
    gamma=0.9995,
    device="cuda",
    ent_coef=0,
    vf_coef=0.5,
    max_grad_norm=10,
    tensorboard_log=log_dir,
)
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])
model.save(log_dir + "/jetbot_policy")

my_env.close()
