# DQNAgent.py
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.DQN.DQNet import DQN
# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('training_data', exist_ok=True)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Neural networks - move to device
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Training parameters
        self.memory = deque(maxlen=20000)  # Increased memory size
        self.batch_size = 128  # Increased batch size for GPU
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998  # Slower decay for more exploration
        self.update_target_every = 100
        
        self.steps = 0
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in eval mode
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            # Random action
            return random.randrange(self.action_size)
        else:
            # Action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return np.argmax(q_values.cpu().numpy())
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert to tensors and move to device
        states = torch.FloatTensor([e[0] for e in batch]).to(device)
        actions = torch.LongTensor([e[1] for e in batch]).to(device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model_safe(self, filename):
        """Save model in safe format"""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'steps': self.steps,
            'device': str(device)
        }
        torch.save(checkpoint, filename, _use_new_zipfile_serialization=True)
        print(f"DQN model saved to {filename}")
    
    def load_model_safe(self, filename):
        """Load model using safe weights_only method"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found")
        
        try:
            checkpoint = torch.load(filename, weights_only=True)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint.get('steps', 0)
            
            # Move models to current device
            self.policy_net.to(device)
            self.target_net.to(device)
            
            print(f"DQN model loaded from {filename}")
            print(f"Model was trained on: {checkpoint.get('device', 'unknown')}")
            print(f"Current device: {device}")
            return True
        except Exception as e:
            print(f"Error loading DQN model: {e}")
            return False

class RobotArmEnvironment:
    def __init__(self, gui=True):
        self.physicsClient = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Load environment
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        
        # Get joint information
        self.numJoints = p.getNumJoints(self.robotId)
        self.joint_indices = list(range(self.numJoints))
        
        # Target position
        self.target_position = None
        self.target_id = None
        self.create_target()
        
        # Set camera
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5])
        
        # Set initial state
        self.reset()
        
        # Step counter
        self.steps = 0
        
    def create_target(self):
        # Create target sphere
        target_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 0.7])
        self.target_id = p.createMultiBody(baseVisualShapeIndex=target_visual, 
                                         basePosition=[0.5, 0, 0.5])
    
    def reset(self):
        # Reset robot to random position
        self.joint_positions = np.array([random.uniform(-1.0, 1.0) for _ in range(self.numJoints)])
        for i in self.joint_indices:
            p.resetJointState(self.robotId, i, self.joint_positions[i])
        
        # Generate random target position
        self.target_position = [
            random.uniform(0.2, 0.8),    # x
            random.uniform(-0.5, 0.5),   # y  
            random.uniform(0.3, 0.7)     # z
        ]
        p.resetBasePositionAndOrientation(self.target_id, self.target_position, [0, 0, 0, 1])
        
        self.steps = 0
        return self.get_state()
    
    def get_state(self):
        # Get current state: joint angles + target position + end effector position
        joint_states = p.getJointStates(self.robotId, self.joint_indices)
        joint_angles = [state[0] for state in joint_states]
        
        # Get end effector position
        end_effector_pos = self.get_end_effector_position()
        
        state = joint_angles + self.target_position + end_effector_pos
        return np.array(state, dtype=np.float32)
    
    def get_end_effector_position(self):
        # Get end effector position
        link_state = p.getLinkState(self.robotId, self.numJoints-1)
        return list(link_state[4])  # World position
    
    def step(self, action_idx):
        # Convert discrete action to continuous joint movements
        action = np.zeros(self.numJoints)
        action[action_idx] = 1.0  # Move only one joint
        
        # Execute action
        self.joint_positions = np.clip(self.joint_positions + action * 0.1, -2.0, 2.0)
        
        for i in self.joint_indices:
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, 
                                  targetPosition=self.joint_positions[i])
        
        # Step simulation
        p.stepSimulation()
        time.sleep(1./240.)
        
        self.steps += 1
        
        # Calculate reward
        reward = self.calculate_reward()
        done = self.is_done()
        
        return self.get_state(), reward, done, {}
    
    def calculate_reward(self):
        # Get end effector position
        end_effector_pos = self.get_end_effector_position()
        
        # Calculate distance to target
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.target_position))
        
        # Distance reward (higher when closer)
        distance_reward = 2.0 / (1.0 + distance)  # Increased reward
        
        # Success reward (if close enough)
        success_reward = 20.0 if distance < 0.05 else 0.0
        
        # Step penalty
        step_penalty = -0.005  # Reduced penalty
        
        return distance_reward + success_reward + step_penalty
    
    def is_done(self):
        end_effector_pos = self.get_end_effector_position()
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.target_position))
        return distance < 0.05 or self.steps > 200
    
    def close(self):
        p.disconnect()

def train_dqn():
    print("=== DQN Reinforcement Learning Training ===")
    print(f"Training on: {device}")
    
    # Create environment
    env = RobotArmEnvironment(gui=False)  # No GUI for faster training
    state_size = 13  # 7 joints + 3 target + 3 end effector
    action_size = 7  # Discrete actions for each joint
    
    agent = DQNAgent(state_size, action_size)
    
    episodes = 1000  # More episodes for better training
    scores = []
    epsilons = []
    losses = []
    
    print(f"Starting DQN training for {episodes} episodes...")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Batch size: {agent.batch_size}, Memory size: {agent.memory.maxlen}")
    
    # Training loop with progress tracking
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_loss = 0
        update_count = 0
        
        while True:
            # Choose action
            action_idx = agent.act(state)
            
            # Execute action
            next_state, reward, done, _ = env.step(action_idx)
            
            # Store experience
            agent.remember(state, action_idx, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train and track loss
            if len(agent.memory) > agent.batch_size:
                agent.replay()
                update_count += 1
            
            if done:
                break
        
        scores.append(total_reward)
        epsilons.append(agent.epsilon)
        
        # Progress reporting
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            success_rate = np.mean([1 if score > 15 else 0 for score in scores[-10:]]) * 100 if len(scores) >= 10 else 0
            
            print(f'Episode: {episode:4d}, Score: {total_reward:6.2f}, '
                  f'Avg Score: {avg_score:6.2f}, Epsilon: {agent.epsilon:.3f}, '
                  f'Steps: {steps:3d}, Success Rate: {success_rate:5.1f}%')
        
        # Save checkpoints
        if episode % 100 == 0:
            agent.save_model_safe(f'models/dqn_model_episode_{episode}.pth')
            print(f"Checkpoint saved at episode {episode}")
    
    # Save final model
    agent.save_model_safe('models/dqn_model_final.pth')
    env.close()
    
    # Plot training results
    plot_training_results(scores, epsilons, episodes)
    
    return scores

def plot_training_results(scores, epsilons, episodes):
    """Plot training results"""
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot scores
    ax1.plot(scores, alpha=0.6, linewidth=0.8, label='Episode Score')
    # Moving average
    window_size = 50
    if len(scores) >= window_size:
        moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(scores)), moving_avg, linewidth=2, 
                label=f'Moving Avg (window={window_size})', color='red')
    
    ax1.set_title('DQN Training Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot epsilon
    ax2.plot(epsilons, color='green', linewidth=2)
    ax2.set_title('Exploration Rate (Epsilon)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training results plot saved to 'dqn_training_results.png'")

def test_dqn_model():
    """Test the trained DQN model"""
    model_file = 'models/dqn_model_final.pth'
    if not os.path.exists(model_file):
        print("DQN model file not found. Please train the model first.")
        return
    
    print("=== Testing DQN Model ===")
    
    # Create environment with GUI
    env = RobotArmEnvironment(gui=True)
    state_size = 13
    action_size = 7
    
    agent = DQNAgent(state_size, action_size)
    
    if not agent.load_model_safe(model_file):
        print("Failed to load DQN model")
        env.close()
        return
    
    print("Model loaded successfully!")
    print(f"Current epsilon: {agent.epsilon}")
    print(f"Using device: {device}")
    
    # Test for several episodes
    test_episodes = 5
    success_count = 0
    total_steps = 0
    
    for episode in range(test_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n--- Test Episode {episode + 1} ---")
        
        while True:
            # Use greedy policy (no exploration)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent.policy_net(state_tensor)
                action_idx = np.argmax(q_values.cpu().numpy())
            
            next_state, reward, done, _ = env.step(action_idx)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Get current distance for info
            end_effector_pos = env.get_end_effector_position()
            distance = np.linalg.norm(np.array(end_effector_pos) - np.array(env.target_position))
            
            if steps % 20 == 0:
                print(f"Step {steps}: Distance = {distance:.3f}, Reward = {reward:.3f}")
            
            if done:
                total_steps += steps
                if distance < 0.05:
                    success_count += 1
                    print(f"SUCCESS! Episode completed in {steps} steps, Total reward: {total_reward:.2f}")
                else:
                    print(f"FAILED! Maximum steps reached. Final distance: {distance:.3f}, Total reward: {total_reward:.2f}")
                break
            
            time.sleep(1./60.)  # Slow down for visualization
    
    success_rate = (success_count / test_episodes) * 100
    avg_steps = total_steps / test_episodes
    print(f"\n=== Test Results ===")
    print(f"Success rate: {success_rate:.1f}% ({success_count}/{test_episodes})")
    print(f"Average steps per episode: {avg_steps:.1f}")
    
    env.close()

def demo_dqn_control():
    """Interactive demo of DQN control"""
    model_file = 'models/dqn_model_final.pth'
    if not os.path.exists(model_file):
        print("DQN model file not found. Please train the model first.")
        return
    
    print("=== DQN Interactive Demo ===")
    print("Controls:")
    print("N: New episode")
    print("Q: Quit")
    
    env = RobotArmEnvironment(gui=True)
    state_size = 13
    action_size = 7
    
    agent = DQNAgent(state_size, action_size)
    
    if not agent.load_model_safe(model_file):
        print("Failed to load DQN model")
        env.close()
        return
    
    state = env.reset()
    
    while True:
        # Use DQN to control the robot
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = agent.policy_net(state_tensor)
            action_idx = np.argmax(q_values.cpu().numpy())
        
        next_state, reward, done, _ = env.step(action_idx)
        state = next_state
        
        # Display info
        end_effector_pos = env.get_end_effector_position()
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(env.target_position))
        
        print(f"Distance: {distance:.3f}, Reward: {reward:.3f}, Action: {action_idx}", end='\r')
        
        if done:
            if distance < 0.05:
                print(f"\nTarget reached! Starting new episode...")
            else:
                print(f"\nEpisode finished. Starting new episode...")
            state = env.reset()
        
        # Check for user input
        keys = p.getKeyboardEvents()
        for key, state_val in keys.items():
            if state_val & p.KEY_WAS_TRIGGERED:
                if key == ord('n'):
                    state = env.reset()
                    print("\nNew episode started!")
                elif key == ord('q'):
                    env.close()
                    return
        
        time.sleep(1./60.)

if __name__ == "__main__":
    print("DQN Robot Arm Control with GPU Support")
    print(f"Available device: {device}")
    print("1: Train DQN model")
    print("2: Test DQN model")
    print("3: Interactive demo")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        train_dqn()
    elif choice == '2':
        test_dqn_model()
    elif choice == '3':
        demo_dqn_control()
    else:
        print("Invalid choice. Starting training...")
        train_dqn()