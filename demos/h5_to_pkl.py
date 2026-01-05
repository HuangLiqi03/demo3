import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # 指定使用GPU
import argparse
import gymnasium as gym
import numpy as np
import torch
import h5py
import json
from tqdm import tqdm
import pickle
from mani_skill.trajectory import utils as trajectory_utils

# Add demo3 to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo3"))

from common.trajectory_saver import BaseTrajectorySaver
from envs import make_env

# Mock Config classes
class MockConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def get(self, key, default=None):
        return getattr(self, key, default)
    def __contains__(self, key):
        return hasattr(self, key)

class ObservationConverter(object):
    def __init__(self, cfg):
        self.obs_flag = cfg.obs != cfg.obs_save
        self.env_obs = None
        self.obs_type = cfg.obs_save
        # We don't support separate env_obs for now in this script
        if self.obs_flag:
             print("Warning: obs != obs_save not fully supported in h5_to_pkl, assuming same env")

    def get_obs(self, env):
        return env.get_obs(self.obs_type)

    def reset(self, task_idx, seed, env):
        # In evaluate.py, this calls env_obs.reset or env.get_obs
        # We assume env is already reset
        return env.get_obs(self.obs_type)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g. ms-pick-cube)")
    parser.add_argument("--traj-path", type=str, required=True, help="Path to the H5 trajectory file")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save the PKL file. Defaults to traj_path with .pkl extension")
    parser.add_argument("--obs-mode", type=str, default="rgb", help="Observation mode (rgb or state)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to convert")
    parser.add_argument("--robot-uids", type=str, default="panda_wristcam", help="Robot UIDs")
    parser.add_argument("--use-env-states", action="store_true", help="Use env states from H5 to force synchronization")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.save_path is None:
        args.save_path = args.traj_path.replace(".h5", ".pkl")
        
    # Load H5
    h5_file = h5py.File(args.traj_path, "r")
    json_path = args.traj_path.replace(".h5", ".json")
    with open(json_path, "r") as f:
        json_data = json.load(f)
    
    episodes = json_data["episodes"]
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
        
    # Construct Config for make_env
    maniskill_cfg = MockConfig(
        camera={"image_size": 128}, # Default in evaluate.py seems to be implied or from config
        sim_backend="gpu", # Use GPU for faster replay if possible, or cpu
        action_penalty=0.0,
        max_bc_steps=3e4,
        action_repeat=1,
        obs_keys=["agent", "image"] if args.obs_mode == "rgb" else ["agent"], # Adjust based on obs_mode
        max_episode_steps=1000 # Default?
    )
    
    cfg = MockConfig(
        task=args.task,
        obs=args.obs_mode,
        obs_save=args.obs_mode,
        num_envs=1,
        maniskill=maniskill_cfg,
        robot_uids=args.robot_uids,
        seed=1,
        max_bc_steps=3e4
    )
    
    # Create Env
    try:
        env = make_env(cfg)
    except Exception as e:
        print(f"Failed to make env: {e}")
        print("Trying to fallback to gym.make if task is not in MANISKILL_TASKS")
        raise e

    # Update max_episode_steps from env
    if hasattr(env.unwrapped, "max_episode_steps"):
        maniskill_cfg.max_episode_steps = env.unwrapped.max_episode_steps
    
    # Saver
    saver = BaseTrajectorySaver(
        num_envs=1,
        save_dir=os.path.dirname(args.save_path),
        success_only=False,
        max_traj=len(episodes)
    )
    
    obs_converter = ObservationConverter(cfg)
    
    print(f"Converting {len(episodes)} episodes...")
    
    for ep in tqdm(episodes):
        episode_id = ep["episode_id"]
        traj_key = f"traj_{episode_id}"
        if traj_key not in h5_file:
            continue
            
        traj_data = h5_file[traj_key]
        actions = traj_data["actions"][:]
        
        # Load env states if requested
        env_states = None
        if args.use_env_states and "env_states" in traj_data:
            env_states = trajectory_utils.dict_to_list_of_dicts(traj_data["env_states"])

        # Reset env
        reset_kwargs = ep["reset_kwargs"]
        seed = ep["episode_seed"]
        
        # Handle seed
        if "seed" in reset_kwargs:
             if isinstance(reset_kwargs["seed"], list):
                 seed = reset_kwargs["seed"][0]
             else:
                 seed = reset_kwargs["seed"]
        
        # Reset
        obs = env.reset(seed=seed) 
        
        # Force initial state if available
        if env_states is not None:
            env.unwrapped.set_state_dict(env_states[0])
            # Re-get obs after setting state
            obs = env.get_obs()

        # Initial transition (NaN action)
        obs_save = obs_converter.get_obs(env)
        
        # Ensure cpu tensors
        def to_cpu(x):
            if hasattr(x, "cpu"):
                return x.cpu()
            if isinstance(x, dict):
                return {k: to_cpu(v) for k, v in x.items()}
            return x

        obs_save = to_cpu(obs_save)
        
        # Initial transition
        # Pass batched tensors directly so saver can index them (e.g. obs_save[0])
        saver.add_transition(
            torch.full_like(env.rand_act(), float("nan")).cpu(),
            obs_save,
            torch.tensor(float("nan")).repeat(1).cpu(),
            torch.tensor(False).repeat(1).cpu(),
            [{}]
        )
        
        for i, action in enumerate(actions):
            # Action from H5 is numpy. Env expects Tensor (due to TensorWrapper).
            action_tensor = torch.from_numpy(action).float().to(env.device).unsqueeze(0) # Add batch dim
            
            step_result = env.step(action_tensor)
            
            # Handle 4 or 5 return values
            if len(step_result) == 5:
                obs, reward, terminated, done, info = step_result
            else:
                obs, reward, done, info = step_result
                terminated = done # Approximation
            
            # Force state synchronization
            if env_states is not None and i + 1 < len(env_states):
                env.unwrapped.set_state_dict(env_states[i+1])
                # Re-get obs after setting state to ensure image matches the forced state
                # Note: reward/done/info are from the step(), which might be slightly different
                # but usually acceptable. Ideally we'd re-calculate them but that's complex.
                # For visual consistency, updating obs is most important.
                obs = env.get_obs()

            # Force done=True on the last step to ensure saving
            if i == len(actions) - 1:
                done = torch.tensor([True], device=done.device)

            obs_save = obs_converter.get_obs(env)
            obs_save = to_cpu(obs_save)
            
            # Convert info to list of dicts to match evaluate.py
            if isinstance(info, dict) or hasattr(info, "keys"):
                 # Handle TensorDict explicitly to avoid iteration bugs
                 if hasattr(info, "to_dict"):
                     info = info.to_dict()
                 
                 if len(info) == 0:
                     info_list = [{}]
                 else:
                     info_list = [dict(zip(info, t)) for t in zip(*info.values())]
            else:
                 info_list = [info]

            saver.add_transition(
                action_tensor.cpu(),
                obs_save,
                reward.cpu(),
                done.cpu(), 
                info_list
            )
            
    # Save
    with open(args.save_path, "wb") as f:
        pickle.dump(saver.data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to {args.save_path}")

if __name__ == "__main__":
    main()


# python demos/h5_to_pkl.py \
#    --task ms-stack-cube-semi \
#    --traj-path /data/huangliqi/demo3/demos/StackCube_DEMO3/motionplanning/20251231_181659.rgb.pd_ee_delta_pose.physx_cpu.h5 \
#    --obs-mode rgb \
#    --robot-uids franka_fr3_wristcam