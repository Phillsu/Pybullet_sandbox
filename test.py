# test_models.py
import time
import numpy as np
from utils import RobotArmEnvironment
from BehaviorCloning import BehaviorCloningModel
from DQN import DQNAgent
import torch
import torch.nn as nn



class MLRobotController:
    def __init__(self, model_type='behavior_cloning'):
        self.env = RobotArmEnvironment(gui=True)
        self.state_size = 13
        self.action_size = 7
        
        if model_type == 'behavior_cloning':
            self.model = BehaviorCloningModel(self.state_size, self.action_size)
            checkpoint = torch.load('models/behavior_cloning_model.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("加载行为克隆模型")
        
        elif model_type == 'dqn':
            self.agent = DQNAgent(self.state_size, self.action_size)
            self.agent.load('models/dqn_model.pth')
            self.agent.policy_net.eval()
            print("加载DQN模型")
        
        self.model_type = model_type
    
    def run_episode(self):
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # 使用模型预测动作
            if self.model_type == 'behavior_cloning':
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = self.model(state_tensor).numpy()[0]
            
            elif self.model_type == 'dqn':
                action_idx = self.agent.act(state)
                action = np.zeros(self.action_size)
                action[action_idx] = 1.0
            
            # 执行动作
            next_state, reward, done, _ = self.env.step(action)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # 获取末端执行器位置
            end_effector_pos = self.env.get_end_effector_position()
            distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.env.target_position))
            
            print(f'Step: {steps}, Reward: {reward:.3f}, Distance: {distance:.3f}')
            
            if done:
                print(f'回合完成! 总奖励: {total_reward:.2f}, 步数: {steps}')
                break
            
            time.sleep(1./60.)  # 控制运行速度
    
    def interactive_test(self):
        """交互式测试模式"""
        print("=== 机器学习模型测试模式 ===")
        print("控制说明:")
        print("N: 新回合")
        print("M: 切换控制模式")
        print("Q: 退出")
        
        current_mode = "AI控制"
        manual_control = False
        
        state = self.env.reset()
        
        while True:
            keys = p.getKeyboardEvents()
            
            for key, state_val in keys.items():
                if state_val & p.KEY_WAS_TRIGGERED:
                    if key == ord('n'):
                        state = self.env.reset()
                        print("新回合开始")
                    
                    elif key == ord('m'):
                        manual_control = not manual_control
                        current_mode = "手动控制" if manual_control else "AI控制"
                        print(f"切换到 {current_mode} 模式")
                    
                    elif key == ord('q'):
                        return
            
            if not manual_control:
                # AI控制
                if self.model_type == 'behavior_cloning':
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        action = self.model(state_tensor).numpy()[0]
                
                elif self.model_type == 'dqn':
                    action_idx = self.agent.act(state)
                    action = np.zeros(self.action_size)
                    action[action_idx] = 1.0
                
                state, reward, done, _ = self.env.step(action)
                
                if done:
                    print("目标达成!")
                    time.sleep(1)
                    state = self.env.reset()
            
            time.sleep(1./60.)
    
    def close(self):
        self.env.close()

def benchmark_models():
    """基准测试不同模型的性能"""
    models = ['behavior_cloning', 'dqn']
    
    for model_name in models:
        print(f"\n=== 测试 {model_name} 模型 ===")
        
        try:
            controller = MLRobotController(model_type=model_name)
            
            # 运行多个测试回合
            successes = 0
            total_rewards = []
            
            for episode in range(10):
                state = controller.env.reset()
                episode_reward = 0
                steps = 0
                
                while steps < 200:  # 最大步数
                    if model_name == 'behavior_cloning':
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        with torch.no_grad():
                            action = controller.model(state_tensor).numpy()[0]
                    else:
                        action_idx = controller.agent.act(state)
                        action = np.zeros(controller.action_size)
                        action[action_idx] = 1.0
                    
                    state, reward, done, _ = controller.env.step(action)
                    episode_reward += reward
                    steps += 1
                    
                    if done:
                        successes += 1
                        break
                
                total_rewards.append(episode_reward)
                print(f'Episode {episode+1}: Reward = {episode_reward:.2f}')
            
            success_rate = successes / 10 * 100
            avg_reward = np.mean(total_rewards)
            
            print(f"成功率: {success_rate:.1f}%")
            print(f"平均奖励: {avg_reward:.2f}")
            
            controller.close()
            
        except Exception as e:
            print(f"测试 {model_name} 时出错: {e}")

if __name__ == "__main__":
    # 测试行为克隆模型
    print("=== 机器学习模型测试 ===")
    
    # 选择要测试的模型
    model_choice = input("选择模型 (1: 行为克隆, 2: DQN, 3: 基准测试): ")
    
    if model_choice == '1':
        controller = MLRobotController(model_type='behavior_cloning')
        controller.interactive_test()
        controller.close()
    
    elif model_choice == '2':
        controller = MLRobotController(model_type='dqn')
        controller.interactive_test()
        controller.close()
    
    elif model_choice == '3':
        benchmark_models()
    
    else:
        print("无效选择")