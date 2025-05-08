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
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        # Remove tensorboard logging to avoid the dependency
        tensorboard_log=None
    )
    
    # Train the agent
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model
    model.save("sac_highway_model")

if __name__ == "__main__":
    train()