import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def create_env():
    # Create and configure the environment before making it
    env = gym.make('highway-v0', render_mode=None)
    
    # Set configuration using env.unwrapped
    env.unwrapped.configure({
        'observation': {
            'type': 'Kinematics',
            'vehicles_count': 5,
            'features': ['presence', 'x', 'y', 'vx', 'vy'],
            'absolute': True
        },
        'action': {
            'type': 'ContinuousAction',
            'longitudinal': True,
            'lateral': True
        },
        'lanes_count': 3,
        'vehicles_count': 15,
        'duration': 40,
        'initial_spacing': 2,
        'simulation_frequency': 15,
        'policy_frequency': 5
    })
    env.reset()
    return Monitor(env)

def train():
    env = DummyVecEnv([create_env])
    
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,  # Reduced learning rate for more stable learning
        buffer_size=200000,  # Increased buffer size
        learning_starts=5000,  # More initial random actions for better exploration
        batch_size=512,  # Larger batch size
        train_freq=2,  # Train every 2 steps
        gradient_steps=2,  # More gradient steps per update
        ent_coef='auto',
        gamma=0.99,  # Added discount factor
        tau=0.02,  # Added soft update coefficient
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256, 128],  # Deeper actor network
                qf=[256, 256, 128]   # Deeper critic network
            )
        ),
        tensorboard_log=None
    )
    
    # Increased training time
    total_timesteps = 300000  # Tripled training time
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model
    model.save("sac_highway_model")

if __name__ == "__main__":
    train()