# 此脚本暂时无用
import h5py
import numpy as np
import pickle
import random
import os
from tqdm import tqdm
def convert_h5_to_pt(h5_path, pt_path):
    """
    将 HDF5 格式的演示数据集转换为 pickle (pt) 格式的轨迹列表。

    Args:
        h5_path (str): HDF5 文件的路径。
        pt_path (str): 转换后 pickle 文件的保存路径。
        
    """
    all_trajectories = []

    print(f"⏳ 正在加载 HDF5 文件: {h5_path}")
    try:
        with h5py.File(h5_path, 'r') as f:
            # 过滤出所有以 'traj_' 开头的键 (即所有轨迹)
            traj_keys = sorted([k for k in f.keys() if k.startswith('traj_')])
            
            print(f"找到 {len(traj_keys)} 条轨迹。")

            for traj_id in tqdm(traj_keys):
                traj_group = f[traj_id]
                
                # 1. 提取 Actions
                actions = traj_group['actions'][()] # [T, A]

                # 2. 提取 Rewards
                # HDF5 文件没有明确的 'rewards' 键。
                # 假设任务成功时奖励为 1，其他情况为 0。
                # 'terminated' 或 'truncated' 标志可以用来确定轨迹长度 T。
                T = actions.shape[0]
                
                # 奖励计算：简化处理，如果 'terminated' 在最后一步为 True，则最后一步奖励为 1，否则为 0。
                # 您可以根据实际的奖励机制调整这一逻辑。
                rewards = np.zeros(T, dtype=np.float32)
                
                # 检查 success 键是否存在
                if 'success' in traj_group:
                    # 如果有 success 键，使用它来定义奖励（例如：最后一步成功的奖励为 1）
                    # 注意: 您文档中的 success/fail 是 [T] 数组，通常RL中奖励是 R_t (执行 a_t 得到)
                    # 假设我们只关心最终状态：
                    if traj_group['success'][()][-1]:
                        rewards[-1] = 1.0 # 假设只有最后一步成功才有奖励
                elif 'terminated' in traj_group and traj_group['terminated'][()][-1]:
                    # 如果没有 success 键，且轨迹在最后一步 'terminated'，则给一个默认奖励
                    rewards[-1] = 1.0
                
                # 3. 提取 Observations/Next_Observations
                # load_dataset_as_td 函数需要 obs 或 next_observations
                # HDF5 文件中的 'obs' 是 [T+1, D]：
                # obs[0:T] 是 O_0, O_1, ..., O_{T-1} (当前观测)
                # obs[1:T+1] 是 O_1, O_2, ..., O_T (下一个观测)
                full_obs = traj_group['obs']['sensor_data']['base_camera']['rgb'][()] # [T+1, D]

                # 满足 load_dataset_as_td 中 'observations' 键的需求:
                # 'observations' (O_0, ..., O_{T-1}) 对应 T 个 (action, reward) 步骤
                observations = full_obs[:-1] # [T, D]
                # 'next_observations' (O_1, ..., O_T) 对应 T 个 (action, reward) 步骤
                next_observations = full_obs[1:] # [T, D]
                
                # 4. 提取 Infos (用于 success_only 过滤)
                terminated = traj_group['terminated'][()] # [T]
                truncated = traj_group['truncated'][()]   # [T]
                
                # 构造 infos 列表：load_dataset_as_td 检查 infos[-1]['success']
                infos = []
                for t in range(T):
                    info = {
                        "terminated": terminated[t].item(),
                        "truncated": truncated[t].item(),
                        # 只有在最后一步，才需要包含 'success' 键供过滤使用
                        "success": (
                            traj_group['success'][()][t].item() 
                            if 'success' in traj_group else False
                        )
                    }
                    infos.append(info)


                # 5. 组合成目标格式的字典
                trajectory_dict = {
                    "actions": actions,                 # [T, A]
                    "rewards": rewards,                 # [T]
                    # 优先使用 next_observations 键，因为它在 RL 中更常用，
                    # 并且您的 load_dataset_as_td 函数会优先读取它。
                    "next_observations": next_observations, # [T, D] 
                    # 您也可以包含 'observations' 
                    "observations": observations,       # [T, D]
                    "dones": terminated | truncated,    # [T] (通常 dones = terminated OR truncated)
                    "infos": infos                      # list of dicts
                }
                
                all_trajectories.append(trajectory_dict)


        print(f"✅ HDF5 文件加载并转换成功，共 {len(all_trajectories)} 条轨迹。")
        print(f"💾 正在保存到 pickle 文件: {pt_path}")

        # 使用 pickle 序列化并保存
        with open(pt_path, "wb") as f:
            pickle.dump(all_trajectories, f)

        print(f"🎉 转换完成！文件已保存至 {pt_path}")

    except Exception as e:
        print(f"❌ 转换过程中发生错误: {e}")


# --- 示例用法 (请替换为您的实际路径) ---
if __name__ == '__main__':
    # 假设您的 HDF5 文件名为 'data.h5'
    H5_FILE_PATH = '/data/huangliqi/.maniskill/demos/StackCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5' 
    # 转换后您想要的 pickle 文件名
    PT_FILE_PATH = '/hard_data/user_dataset/huangliqi_dataset/demo3/StackCube.pt'

    if os.path.exists(H5_FILE_PATH):
        convert_h5_to_pt(H5_FILE_PATH, PT_FILE_PATH)
    else:
        print(f"错误：未找到文件 {H5_FILE_PATH}")
    
    print("请手动将上面的路径替换为您的实际文件路径并取消注释 main 块来运行脚本。")