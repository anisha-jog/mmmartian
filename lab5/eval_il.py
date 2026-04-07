import os
import torch
import numpy as np
import gymnasium as gym
from rl import TouchEnv
from imitation import ImitationPolicy

# Setup Env
env = gym.make('TouchEnv', render_mode='human')

# Load Policy (weights saved from imitation.py; avoids pickle looking up __main__.ImitationPolicy)
_base = os.path.dirname(os.path.abspath(__file__))
_ckpt = torch.load(os.path.join(_base, "imitation_policy.pt"), weights_only=False)
policy = ImitationPolicy(_ckpt["obs_dim"], _ckpt["act_dim"])
policy.load_state_dict(_ckpt["state_dict"])
policy.eval()

obs, info = env.reset(seed=np.random.randint(9000000))

terminated = False
truncated = False
while not terminated and not truncated:
    with torch.no_grad():
        action = policy(torch.Tensor(obs))

    obs, reward, terminated, truncated, info = env.step(action)
    # print(reward)

