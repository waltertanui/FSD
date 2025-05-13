import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def create_env():
    env = gym.make('highway-v0', render_mode=None) # Added this line, was missing in the provided snippet
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
        'policy_frequency': 5,
        'reward_speed_range': [22, 30],  
        'collision_reward': -5,
        'right_lane_reward': 0.1,
        'high_speed_reward': 0.5, 
    })
    env.reset()
    return Monitor(env)

def train():
    env = DummyVecEnv([create_env])
    
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=200000,
        learning_starts=5000,
        batch_size=512,
        train_freq=2,
        gradient_steps=2,
        ent_coef='auto',
        gamma=0.99,
        tau=0.02,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256, 128],
                qf=[256, 256, 128]
            )
        ),
        tensorboard_log=None
    )
    
    # Adjust total timesteps for 3500 episodes
    timesteps_per_episode = 15 * 40  # 15 Hz * 40 seconds
    total_timesteps = 3500 * timesteps_per_episode
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model
    model.save("sac_highway_model")

if __name__ == "__main__":
    train()