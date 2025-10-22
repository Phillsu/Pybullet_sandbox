# BehaviorCloning.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import matplotlib
# Use non-interactive backend for WSL
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from models.BehaviorClone.BehaviorCloningModel import BehaviorCloningModel

# Create model directory
os.makedirs('models', exist_ok=True)

class BehaviorCloningTrainer:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = BehaviorCloningModel(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        
    def load_demonstrations(self, filename):
        """Load demonstration data"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Demonstration file {filename} not found")
            
        with open(filename, 'rb') as f:
            demonstrations = pickle.load(f)
        
        states = []
        actions = []
        
        print(f"Loaded {len(demonstrations)} demonstrations")
        
        for episode_idx, episode in enumerate(demonstrations):
            for state, action, _, _, _ in episode:
                states.append(state)
                actions.append(action)
            
            if (episode_idx + 1) % 10 == 0:
                print(f"Processed {episode_idx + 1} demonstrations")
        
        states = np.array(states)
        actions = np.array(actions)
        
        print(f"Total samples: {len(states)}")
        print(f"State dimension: {states.shape}")
        print(f"Action dimension: {actions.shape}")
        
        return states, actions
    
    def preprocess_data(self, states, actions):
        """Preprocess data"""
        # Normalize state data
        state_mean = np.mean(states, axis=0)
        state_std = np.std(states, axis=0) + 1e-8  # Avoid division by zero
        states_normalized = (states - state_mean) / state_std
        
        # Action data is already in [-1, 1] range, no need to normalize
        
        return states_normalized, state_mean, state_std
    
    def train(self, states, actions, epochs=100, batch_size=32, validation_split=0.2):
        """Train the model"""
        # Data preprocessing
        states_normalized, state_mean, state_std = self.preprocess_data(states, actions)
        
        # Save normalization parameters
        self.state_mean = state_mean
        self.state_std = state_std
        
        # Split into training and validation sets
        dataset_size = len(states_normalized)
        indices = np.random.permutation(dataset_size)
        split_idx = int(dataset_size * (1 - validation_split))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(states_normalized[train_indices]),
            torch.FloatTensor(actions[train_indices])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(states_normalized[val_indices]),
            torch.FloatTensor(actions[val_indices])
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_states, batch_actions in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                predicted_actions = self.model(batch_states)
                
                # Calculate loss
                loss = self.criterion(predicted_actions, batch_actions)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_states, batch_actions in val_loader:
                    predicted_actions = self.model(batch_states)
                    loss = self.criterion(predicted_actions, batch_actions)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            self.train_losses.append((avg_train_loss, avg_val_loss))
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model_safe('models/behavior_cloning_best.pth')
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}]')
                print(f'  Training Loss: {avg_train_loss:.4f}')
                print(f'  Validation Loss: {avg_val_loss:.4f}')
                print(f'  Best Validation Loss: {best_val_loss:.4f}')
                print('-' * 50)
        
        # Save final model
        self.save_model_safe('models/behavior_cloning_final.pth')
        
        return self.train_losses
    
    def plot_training_history(self):
        """Plot training history and save to file (no display in WSL)"""
        if not self.train_losses:
            print("No training history data")
            return
        
        train_losses = [loss[0] for loss in self.train_losses]
        val_losses = [loss[1] for loss in self.train_losses]
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Behavior Cloning Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save the plot to file
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("Training history plot saved to 'training_history.png'")
        
        # Close the plot to free memory
        plt.close()
    
    def save_model_safe(self, filename):
        """Save model in a safe format that can be loaded with weights_only=True"""
        # Save only the model state dict and necessary parameters
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'state_mean': self.state_mean.tolist() if hasattr(self.state_mean, 'tolist') else self.state_mean,
            'state_std': self.state_std.tolist() if hasattr(self.state_std, 'tolist') else self.state_std,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'train_losses': self.train_losses
        }
        torch.save(checkpoint, filename, _use_new_zipfile_serialization=True)
        print(f"Model saved to {filename} (safe format)")
    
    def load_model_safe(self, filename):
        """Load model using safe weights_only method"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found")
        
        try:
            # Try loading with weights_only=True (safe mode)
            checkpoint = torch.load(filename, weights_only=True)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Convert lists back to numpy arrays if needed
            if isinstance(checkpoint['state_mean'], list):
                self.state_mean = np.array(checkpoint['state_mean'])
            else:
                self.state_mean = checkpoint['state_mean']
                
            if isinstance(checkpoint['state_std'], list):
                self.state_std = np.array(checkpoint['state_std'])
            else:
                self.state_std = checkpoint['state_std']
                
            self.train_losses = checkpoint.get('train_losses', [])
            
            print(f"Model loaded from {filename} (safe mode)")
            return True
            
        except Exception as e:
            print(f"Error loading model with safe method: {e}")
            return False
    
    def save_model(self, filename):
        """Legacy save method - use save_model_safe instead"""
        self.save_model_safe(filename)
    
    def load_model(self, filename):
        """Legacy load method - use load_model_safe instead"""
        return self.load_model_safe(filename)

def analyze_demonstration_data(filename):
    """Analyze demonstration data and save plot to file"""
    if not os.path.exists(filename):
        print(f"File {filename} does not exist")
        return
    
    with open(filename, 'rb') as f:
        demonstrations = pickle.load(f)
    
    print("=== Demonstration Data Analysis ===")
    print(f"Number of demonstrations: {len(demonstrations)}")
    
    # Count steps per demonstration
    episode_lengths = [len(episode) for episode in demonstrations]
    print(f"Average steps: {np.mean(episode_lengths):.1f}")
    print(f"Minimum steps: {np.min(episode_lengths)}")
    print(f"Maximum steps: {np.max(episode_lengths)}")
    
    # Analyze action range
    all_actions = []
    for episode in demonstrations:
        for _, action, _, _, _ in episode:
            all_actions.append(action)
    
    all_actions = np.array(all_actions)
    print(f"Action range: [{np.min(all_actions):.3f}, {np.max(all_actions):.3f}]")
    print(f"Action mean: {np.mean(all_actions, axis=0)}")
    print(f"Action std: {np.std(all_actions, axis=0)}")
    
    # Plot demonstration length distribution and save to file
    plt.figure(figsize=(10, 6))
    plt.hist(episode_lengths, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Demonstration Length Distribution')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('demo_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Demonstration length distribution plot saved to 'demo_length_distribution.png'")

def train_behavior_cloning():
    print("=== Behavior Cloning Training ===")
    
    # Check if data file exists
    data_file = 'training_data/demonstrations.pkl'
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} does not exist")
        print("Please run data collection program first to generate demonstration data")
        return
    
    # Analyze data
    analyze_demonstration_data(data_file)
    
    # State dimension: 7 joint angles + 3 target positions + 3 end effector positions = 13
    # Action dimension: 7 joints
    state_size = 13
    action_size = 7
    
    # Create trainer
    trainer = BehaviorCloningTrainer(state_size, action_size)
    
    try:
        # Load demonstration data
        print("\n=== Loading Demonstration Data ===")
        states, actions = trainer.load_demonstrations(data_file)
        
        # Train model
        print("\n=== Starting Training ===")
        losses = trainer.train(states, actions, epochs=50, batch_size=64)  # Reduced epochs for testing
        
        # Plot training history
        print("\n=== Plotting Training History ===")
        trainer.plot_training_history()
        
        print("\n=== Training Completed! ===")
        print("Model files saved to 'models/' directory")
        print("Training plot saved to 'training_history.png'")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

def test_loaded_model():
    """Test loaded model"""
    model_file = 'models/behavior_cloning_best.pth'
    if not os.path.exists(model_file):
        print("Model file does not exist, please train the model first")
        print("Run with option 1 to train a new model")
        print("Or run with option 4 to create a test model")
        return
    
    # State dimension: 7 joint angles + 3 target positions + 3 end effector positions = 13
    # Action dimension: 7 joints
    state_size = 13
    action_size = 7
    
    trainer = BehaviorCloningTrainer(state_size, action_size)
    
    if not trainer.load_model_safe(model_file):
        print("Failed to load model with safe method")
        print("The model file might be corrupted or in an incompatible format")
        print("Please train a new model")
        return
    
    # Create a test state
    test_state = np.random.randn(state_size).astype(np.float32)
    
    # Normalize state
    test_state_normalized = (test_state - trainer.state_mean) / trainer.state_std
    test_state_tensor = torch.FloatTensor(test_state_normalized).unsqueeze(0)
    
    # Predict action
    trainer.model.eval()
    with torch.no_grad():
        predicted_action = trainer.model(test_state_tensor).numpy()[0]
    
    print("\n=== Model Test Successful ===")
    print("Test State Prediction:")
    print(f"Input state shape: {test_state.shape}")
    print(f"Predicted action shape: {predicted_action.shape}")
    print(f"Action range: [{np.min(predicted_action):.3f}, {np.max(predicted_action):.3f}]")
    print(f"Action values: {predicted_action}")

def create_test_model():
    """Create a simple test model for debugging"""
    state_size = 13
    action_size = 7
    
    trainer = BehaviorCloningTrainer(state_size, action_size)
    
    # Create some dummy data and train briefly
    print("Creating test model with dummy data...")
    dummy_states = np.random.randn(100, state_size).astype(np.float32)
    dummy_actions = np.random.uniform(-1, 1, (100, action_size)).astype(np.float32)
    
    # Train for just a few epochs
    trainer.train(dummy_states, dummy_actions, epochs=3, batch_size=10)
    print("Test model created successfully!")
    print("You can now test model loading with option 2")

def check_environment():
    """Check the current environment and dependencies"""
    print("=== Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    print("Environment: WSL (non-interactive mode)")
    print("Plots will be saved as image files")

def list_model_files():
    """List all model files in the models directory"""
    print("=== Model Files ===")
    if os.path.exists('models'):
        files = os.listdir('models')
        if files:
            for file in files:
                file_path = os.path.join('models', file)
                file_size = os.path.getsize(file_path)
                print(f"  {file} ({file_size} bytes)")
        else:
            print("  No model files found")
    else:
        print("  models directory does not exist")

if __name__ == "__main__":
    # Display environment info
    check_environment()
    print()
    
    # List available model files
    list_model_files()
    print()
    
    # Choose to run training or testing
    print("Select mode:")
    print("1: Train model (requires demonstration data)")
    print("2: Test model")
    print("3: Analyze data") 
    print("4: Create test model (quick test)")
    print("5: Environment check")
    
    choice = input("Enter your choice (1-5): ")
    
    if choice == '1':
        train_behavior_cloning()
    elif choice == '2':
        test_loaded_model()
    elif choice == '3':
        analyze_demonstration_data('training_data/demonstrations.pkl')
    elif choice == '4':
        create_test_model()
    elif choice == '5':
        check_environment()
    else:
        print("Invalid choice, running training mode")
        train_behavior_cloning()