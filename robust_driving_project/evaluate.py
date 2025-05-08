import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from ensemble_q_wrapper import EnsembleWrapper
from fallback_policy import FallbackPolicy
from typing import Dict, List
import json
from stable_baselines3.common.vec_env import DummyVecEnv

class Evaluator:
    def __init__(self, model_path: str):
        # Wrap the environment with DummyVecEnv
        self.env = DummyVecEnv([lambda: gym.make('highway-v0')])
        self.model = SAC.load(model_path)
        self.ensemble_wrapper = EnsembleWrapper(self.model.policy)
        self.fallback_policy = FallbackPolicy()
        
        self.metrics = {
            'success_rate': 0,
            'average_speed': [],
            'fallback_triggers': 0,
            'episode_lengths': []
        }
    
    def evaluate(self, n_episodes: int = 10):
        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_speed = []
            episode_length = 0
            
            while not done:
                use_fallback, uncertainty = self.ensemble_wrapper.should_use_fallback(obs)
                
                if use_fallback:
                    action, _ = self.fallback_policy.predict(obs)
                    self.metrics['fallback_triggers'] += 1
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                
                obs, reward, done, info = self.env.step(action)
                episode_speed.append(info.get('speed', 0))
                episode_length += 1
            
            self.metrics['success_rate'] += int(info.get('is_success', 0))
            self.metrics['average_speed'].append(np.mean(episode_speed))
            self.metrics['episode_lengths'].append(episode_length)
        
        self._summarize_results(n_episodes)
    
    def _summarize_results(self, n_episodes: int):
        summary = {
            'success_rate': self.metrics['success_rate'] / n_episodes * 100,
            'average_speed': np.mean(self.metrics['average_speed']),
            'total_fallbacks': self.metrics['fallback_triggers'],
            'avg_episode_length': np.mean(self.metrics['episode_lengths'])
        }
        
        print("\nEvaluation Summary:")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Average Speed: {summary['average_speed']:.2f} m/s")
        print(f"Total Fallback Triggers: {summary['total_fallbacks']}")
        print(f"Average Episode Length: {summary['avg_episode_length']:.2f}")
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(summary, f, indent=4)

if __name__ == "__main__":
    evaluator = Evaluator("sac_highway_model")
    evaluator.evaluate(n_episodes=10)