# 确定奖励是否合法
import pickle
import numpy as np
import torch

path = '/hard_data/user_dataset/huangliqi_dataset/demo3/demos/ms-place-sphere-semi-fr3/ms-place-sphere-semi_trajectories_20.pkl'

try:
    with open(path, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    all_rewards = []
    for i, traj in enumerate(trajectories):
        rewards = traj['rewards']
        all_rewards.append(rewards)
        # print(f"Traj {i} max reward: {np.max(rewards)}")
        
    all_rewards = np.concatenate(all_rewards)
    print(f"Max reward in dataset: {np.max(all_rewards)}")
    print(f"Min reward in dataset: {np.min(all_rewards)}")
    print(f"Unique rewards: {np.unique(all_rewards)}")
    
except Exception as e:
    print(f"Error reading pkl: {e}")
