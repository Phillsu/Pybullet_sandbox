import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import random
import pickle
import os
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 创建数据目录
os.makedirs('training_data', exist_ok=True)
os.makedirs('models', exist_ok=True)

class RobotArmEnvironment:
    def __init__(self, gui=True):
        self.physicsClient = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # 加载环境
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        
        # 获取关节信息
        self.numJoints = p.getNumJoints(self.robotId)
        self.joint_indices = list(range(self.numJoints))
        
        # 目标位置
        self.target_position = None
        self.target_id = None
        self.create_target()
        
        # 设置相机
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5])
        
        # 设置初始状态
        self.reset()
        
        # 步数计数器
        self.steps = 0
        
    def create_target(self):
        # 创建目标球体
        target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 0.7])
        self.target_id = p.createMultiBody(baseVisualShapeIndex=target_visual, 
                                         basePosition=[0.5, 0, 0.5])
    
    def reset(self):
        # 重置机器人到随机位置
        self.joint_positions = np.array([random.uniform(-1.0, 1.0) for _ in range(self.numJoints)])
        for i in self.joint_indices:
            p.resetJointState(self.robotId, i, self.joint_positions[i])
        
        # 随机生成目标位置
        self.target_position = [
            random.uniform(0.2, 0.8),    # x
            random.uniform(-0.5, 0.5),   # y  
            random.uniform(0.3, 0.7)     # z
        ]
        p.resetBasePositionAndOrientation(self.target_id, self.target_position, [0, 0, 0, 1])
        
        self.steps = 0
        return self.get_state()
    
    def get_state(self):
        # 获取当前状态：关节角度 + 目标位置 + 末端位置
        joint_states = p.getJointStates(self.robotId, self.joint_indices)
        joint_angles = [state[0] for state in joint_states]
        
        # 获取末端执行器位置
        end_effector_pos = self.get_end_effector_position()
        
        state = joint_angles + self.target_position + end_effector_pos
        return np.array(state, dtype=np.float32)
    
    def get_end_effector_position(self):
        # 获取末端执行器位置
        link_state = p.getLinkState(self.robotId, self.numJoints-1)
        return list(link_state[4])  # 世界坐标系位置
    
    def step(self, action):
        # 确保 action 是 numpy array
        action = np.array(action, dtype=np.float32)
        
        # 执行动作
        self.joint_positions = np.clip(self.joint_positions + action * 0.1, -2.0, 2.0)
        
        for i in self.joint_indices:
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, 
                                  targetPosition=self.joint_positions[i])
        
        # 步进仿真
        p.stepSimulation()
        time.sleep(1./240.)
        
        self.steps += 1
        
        # 计算奖励
        reward = self.calculate_reward()
        done = self.is_done()
        
        return self.get_state(), reward, done, {}
    
    def calculate_reward(self):
        # 获取末端执行器位置
        end_effector_pos = self.get_end_effector_position()
        
        # 计算到目标的距离
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.target_position))
        
        # 距离奖励（越近奖励越高）
        distance_reward = 1.0 / (1.0 + distance)
        
        # 成功奖励（如果距离足够近）
        success_reward = 10.0 if distance < 0.05 else 0.0
        
        # 动作惩罚（鼓励平滑运动）
        action_penalty = -0.01
        
        return distance_reward + success_reward + action_penalty
    
    def is_done(self):
        end_effector_pos = self.get_end_effector_position()
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.target_position))
        return distance < 0.05 or self.steps > 200
    
    def close(self):
        p.disconnect()

class DemonstrationCollector:
    def __init__(self, env):
        self.env = env
        self.demo_data = []
        self.selected_joint = 0  # 初始化选中的关节
        
    def collect_demonstration(self):
        print("开始收集演示数据...")
        print("使用以下控制:")
        print("1-7: 选择关节")
        print("W/S: 增加/减少关节角度") 
        print("R: 重置环境")
        print("D: 保存演示")
        print("Q: 退出")
        
        state = self.env.reset()
        episode_data = []
        steps = 0
        
        while True:
            keys = p.getKeyboardEvents()
            action_taken = False
            action = [0.0] * self.env.numJoints
            
            for key, state_val in keys.items():
                if state_val & p.KEY_WAS_TRIGGERED:
                    if key >= ord('1') and key <= ord('7'):
                        joint_index = key - ord('1')
                        if joint_index < self.env.numJoints:
                            self.selected_joint = joint_index
                            print(f"选择关节 {self.selected_joint + 1}")
                    
                    elif key == ord('w'):
                        action[self.selected_joint] = 1.0
                        action_taken = True
                    
                    elif key == ord('s'):
                        action[self.selected_joint] = -1.0
                        action_taken = True
                    
                    elif key == ord('r'):
                        state = self.env.reset()
                        episode_data = []
                        steps = 0
                        print("环境重置")
                    
                    elif key == ord('d'):
                        # 保存演示
                        if len(episode_data) > 0:
                            self.demo_data.append(episode_data)
                            print(f"演示保存，当前演示数量: {len(self.demo_data)}")
                            episode_data = []
                    
                    elif key == ord('q'):
                        return
            
            if action_taken:
                next_state, reward, done, _ = self.env.step(action)
                episode_data.append((state, action, reward, next_state, done))
                state = next_state
                steps += 1
                
                # 显示当前状态信息
                end_effector_pos = self.env.get_end_effector_position()
                distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.env.target_position))
                print(f"步数: {steps}, 距离: {distance:.3f}, 奖励: {reward:.3f}")
                
                if done:
                    print("目标达成! 开始新回合")
                    state = self.env.reset()
                    self.demo_data.append(episode_data)
                    episode_data = []
                    steps = 0
                    print(f"回合完成，演示数量: {len(self.demo_data)}")
            
            time.sleep(1./60.)
    
    def save_demonstrations(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.demo_data, f)
        print(f"演示数据保存到 {filename}, 总演示数: {len(self.demo_data)}")
    
    def load_demonstrations(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.demo_data = pickle.load(f)
            print(f"从 {filename} 加载演示数据, 演示数: {len(self.demo_data)}")
        else:
            print(f"文件 {filename} 不存在")

class BehaviorCloningModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(BehaviorCloningModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))  # 输出在 [-1, 1] 范围内

# 数据集生成主程序
def generate_training_data():
    print("=== 机械手臂训练数据生成 ===")
    env = RobotArmEnvironment(gui=True)
    collector = DemonstrationCollector(env)
    
    try:
        collector.collect_demonstration()
    except KeyboardInterrupt:
        print("\n数据收集中断")
    
    # 保存数据
    if len(collector.demo_data) > 0:
        collector.save_demonstrations('training_data/demonstrations.pkl')
        print(f"成功收集 {len(collector.demo_data)} 个演示")
        
        # 显示数据统计
        total_steps = sum(len(episode) for episode in collector.demo_data)
        print(f"总步数: {total_steps}")
    else:
        print("没有收集到演示数据")
    
    env.close()

# 简单的测试函数
def test_environment():
    """测试环境是否正常工作"""
    print("=== 测试环境 ===")
    env = RobotArmEnvironment(gui=True)
    
    print("环境测试中...按1-7选择关节，W/S移动关节，R重置，Q退出")
    
    selected_joint = 0
    joint_positions = np.array([0.0] * env.numJoints)
    
    while True:
        keys = p.getKeyboardEvents()
        
        for key, state_val in keys.items():
            if state_val & p.KEY_WAS_TRIGGERED:
                if key >= ord('1') and key <= ord('7'):
                    joint_index = key - ord('1')
                    if joint_index < env.numJoints:
                        selected_joint = joint_index
                        print(f"选择关节 {selected_joint + 1}")
                
                elif key == ord('w'):
                    joint_positions[selected_joint] += 0.1
                    p.setJointMotorControl2(env.robotId, selected_joint, p.POSITION_CONTROL, 
                                          targetPosition=joint_positions[selected_joint])
                    print(f"关节 {selected_joint + 1} 角度: {joint_positions[selected_joint]:.2f}")
                
                elif key == ord('s'):
                    joint_positions[selected_joint] -= 0.1
                    p.setJointMotorControl2(env.robotId, selected_joint, p.POSITION_CONTROL, 
                                          targetPosition=joint_positions[selected_joint])
                    print(f"关节 {selected_joint + 1} 角度: {joint_positions[selected_joint]:.2f}")
                
                elif key == ord('r'):
                    env.reset()
                    joint_positions = np.array([0.0] * env.numJoints)
                    print("环境重置")
                
                elif key == ord('q'):
                    env.close()
                    return
        
        p.stepSimulation()
        time.sleep(1./240.)

if __name__ == "__main__":
    # 可以选择运行测试或数据收集
    choice = input("选择模式 (1: 测试环境, 2: 收集数据): ")
    
    if choice == '1':
        test_environment()
    elif choice == '2':
        generate_training_data()
    else:
        print("无效选择，运行数据收集模式")
        generate_training_data()