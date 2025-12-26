import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
import os
import numpy as np
import torch
import imageio

# Create the environment with render_mode="rgb_array" for server-side rendering
env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="rgbd", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="rgb_array" ,# Changed from "human" to "rgb_array"
    robot_uids = 'franka_fr3_wristcam'
)

# Wrap the environment to record videos
# Videos will be saved in the "videos" directory
env = RecordEpisode(
    env,
    output_dir="videos",
    save_trajectory=False,
    save_video=True,
    info_on_video=True,
    video_fps=30
)

print("Observation space", env.observation_space)
print("Action space", env.action_space)

# Create sensor_data directory
os.makedirs("sensor_data", exist_ok=True)

obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
step = 0
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Save camera images
    if "sensor_data" in obs:
        for cam_name, cam_data in obs["sensor_data"].items():
            if "rgb" in cam_data:
                rgb = cam_data["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                
                # rgb is (B, H, W, 4) or (B, H, W, 3)
                img = rgb[0] # Take first env
                
                # Convert to uint8
                if img.dtype in [np.float32, np.float64]:
                    img = (img * 255).astype(np.uint8)
                
                # Remove alpha if present
                if img.shape[-1] == 4:
                    img = img[..., :3]
                    
                imageio.imwrite(f"sensor_data/{step}_{cam_name}.png", img)

    # breakpoint()
    step += 1
    done = terminated or truncated
    # env.render() # No longer needed, RecordEpisode handles video saving
env.close()
print("Video saved to videos/ directory")
print("Images saved to sensor_data/ directory")
