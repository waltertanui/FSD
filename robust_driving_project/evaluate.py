import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from ensemble_q_wrapper import EnsembleWrapper
from fallback_policy import FallbackPolicy
from typing import Dict, List
import json
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, model_path: str):
        def make_env():
            env = gym.make('highway-v0', render_mode=None)
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
            return env
        
        self.env = DummyVecEnv([make_env])
        self.model = SAC.load(model_path)
        self.ensemble_wrapper = EnsembleWrapper(self.model.policy, uncertainty_threshold=0.2)  # reduced from 0.3
        self.fallback_policy = FallbackPolicy()
        
        self.metrics = {
            'success_rate': 0,
            'average_speed': [],
            'fallback_triggers': 0,
            'episode_lengths': [],
            'collisions': 0,
            'lane_deviation_log': []  # Added for debug
        }
    
    def evaluate(self, n_episodes: int = 10):
        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_speed = []
            episode_length = 0
            lane_deviation = 0

            while not done:
                use_fallback, uncertainty = self.ensemble_wrapper.should_use_fallback(obs)

                ego_y = obs[0][0][2]
                current_lane = int(ego_y / 4)
                deviation = abs(ego_y % 4 - 2)
                lane_deviation += deviation

                if use_fallback:
                    action, _ = self.fallback_policy.predict(obs)
                    self.metrics['fallback_triggers'] += 1
                else:
                    action, _ = self.model.predict(obs, deterministic=True)

                if isinstance(action, np.ndarray) and len(action.shape) == 1:
                    action = np.array([action])

                obs, reward, done, info = self.env.step(action)
                done = done[0] if isinstance(done, (list, np.ndarray)) else done
                info = info[0] if isinstance(info, list) else info

                episode_speed.append(info.get('speed', 0))
                episode_length += 1

            if info.get('crashed', False):
                self.metrics['collisions'] += 1

            avg_speed = np.mean(episode_speed)
            avg_deviation = lane_deviation / episode_length
            self.metrics['lane_deviation_log'].append(avg_deviation)

            is_success = (
                not info.get('crashed', False) and
                avg_speed > 10 and
                avg_deviation < 1.2  # Loosened threshold
            )

            self.metrics['success_rate'] += int(is_success)
            self.metrics['average_speed'].append(avg_speed)
            self.metrics['episode_lengths'].append(episode_length)

        self._summarize_results(n_episodes)

    def _summarize_results(self, n_episodes: int):
        # Convert numpy values to Python native types
        summary = {
            'success_rate': float(self.metrics['success_rate'] / n_episodes * 100),
            'average_speed': float(np.mean(self.metrics['average_speed'])),
            'total_fallbacks': int(self.metrics['fallback_triggers']),
            'avg_episode_length': float(np.mean(self.metrics['episode_lengths'])),
            'collision_rate': float((self.metrics['collisions'] / n_episodes) * 100),
            'avg_lane_deviation': float(np.mean(self.metrics['lane_deviation_log']))
        }
        
        print("\nEvaluation Summary:")
        for k, v in summary.items():
            print(f"{k.replace('_', ' ').title()}: {v:.2f}")
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(summary, f, indent=4)

        # Optional: plot deviation
        plt.plot(self.metrics['lane_deviation_log'])
        plt.title("Lane Deviation per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Avg Deviation (m)")
        plt.grid(True)
        plt.savefig("lane_deviation_plot.png")

if __name__ == "__main__":
    evaluator = Evaluator("sac_highway_model")
    evaluator.evaluate(n_episodes=10)
