import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from ensemble_q_wrapper import EnsembleWrapper
from fallback_policy import FallbackPolicy
from typing import Dict, List
import json
from stable_baselines3.common.vec_env import DummyVecEnv
# Add this near the top with other imports
import os
import matplotlib.pyplot as plt

# Add this at the top with other imports
COLOR_PALETTE = {
    'safe': '#4daf4a',       # Green
    'threshold': '#984ea3',  # Purple
    'intervention': '#ff7f00', # Orange
    'baseline': '#377eb8',    # Blue
    'failure': '#e41a1c'      # Red
}

class Evaluator:
    def __init__(self, model_path: str):
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        # Define make_env function for environment creation
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
                'simulation_frequency': 20,  # Updated to 20 Hz
                'policy_frequency': 10,      # Updated to 10 Hz
                'max_time_step': 800,        # Added max_time_step
                'duration': 40,              # Set maximum duration to 40 seconds
            })
            env.reset()
            return env
        
        # Initialize the environment
        self.env = DummyVecEnv([make_env])
        
        # Load model FIRST
        self.model = SAC.load(model_path)
        
        # THEN create ensemble wrapper
        # Lower the threshold further to ensure fallback triggers based on new avg uncertainty
        self.ensemble_wrapper = EnsembleWrapper(self.model.policy, uncertainty_threshold=0.02) # Was 0.04
        
        # Rest of initialization
        self.fallback_policy = FallbackPolicy()
        self.metrics = {
            'success_rate': 0,
            'average_speed': [],
            'fallback_triggers': 0, # Total fallback triggers across all steps
            'episode_lengths': [],
            'collisions': 0,
            'lane_deviation_log': [],
            'episode_rewards': [],
            'collected_uncertainties': [], # Uncertainties when RL agent is active
            'total_rl_steps': 0,
            'episode_step_uncertainties': [], # List of lists: uncertainty at each step for each episode
            'episode_step_fallbacks': []      # List of lists: fallback active (1/0) at each step for each episode
        }
    
    def evaluate(self, n_episodes: int = 10):
        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_speed = []
            episode_length = 0
            lane_deviation = 0
            current_episode_reward = 0.0
            
            # For detailed per-step logging within an episode
            current_episode_step_uncertainties = []
            current_episode_step_fallbacks = []

            while not done:
                use_fallback, uncertainty = self.ensemble_wrapper.should_use_fallback(obs)
                
                # Log per-step data
                current_episode_step_uncertainties.append(uncertainty)
                current_episode_step_fallbacks.append(1 if use_fallback else 0)

                ego_y = obs[0][0][2] # Assuming obs structure [batch_idx, vehicle_idx, feature_idx]
                current_lane = int(ego_y / 4)
                deviation = abs(ego_y % 4 - 2)
                lane_deviation += deviation

                if use_fallback:
                    action, _ = self.fallback_policy.predict(obs)
                    # fallback_triggers is already incremented globally per step in the previous version
                    # No need to increment self.metrics['fallback_triggers'] here if it's a global step count
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                    self.metrics['collected_uncertainties'].append(uncertainty)
                    self.metrics['total_rl_steps'] += 1
                
                if isinstance(action, np.ndarray) and len(action.shape) == 1:
                    action = np.array([action])

                obs, reward, done, info = self.env.step(action)
                done = done[0] if isinstance(done, (list, np.ndarray)) else done
                info = info[0] if isinstance(info, list) else info
                current_episode_reward += reward[0] 

                episode_speed.append(info.get('speed', 0))
                episode_length += 1

            if info.get('crashed', False):
                self.metrics['collisions'] += 1

            avg_speed = np.mean(episode_speed)
            avg_deviation = lane_deviation / episode_length
            self.metrics['lane_deviation_log'].append(avg_deviation)

            is_success = (
                not info.get('crashed', False) and
                avg_speed > 8 and  # Reduced from 10
                avg_deviation < 1.5  # Increased from 1.2
            )

            self.metrics['success_rate'] += int(is_success)
            self.metrics['average_speed'].append(avg_speed)
            self.metrics['episode_lengths'].append(episode_length)
            self.metrics['episode_rewards'].append(current_episode_reward)
            self.metrics['episode_step_uncertainties'].append(current_episode_step_uncertainties)
            self.metrics['episode_step_fallbacks'].append(current_episode_step_fallbacks)

        self._summarize_results(n_episodes)

    # Modify the plotting sections in _summarize_results
    def _summarize_results(self, n_episodes: int):
        # Calculate additional metrics
        avg_reward = np.mean(self.metrics['episode_rewards']) if self.metrics['episode_rewards'] else 0.0
        avg_uncertainty_rl_active = np.mean(self.metrics['collected_uncertainties']) if self.metrics['collected_uncertainties'] else 0.0
        
        total_steps_all_episodes = np.sum(self.metrics['episode_lengths'])
        fallback_activation_rate_percent = (self.metrics['fallback_triggers'] / total_steps_all_episodes * 100) if total_steps_all_episodes > 0 else 0.0

        summary = {
            'success_rate': float(self.metrics['success_rate'] / n_episodes * 100),
            'average_speed': float(np.mean(self.metrics['average_speed']) if self.metrics['average_speed'] else 0.0),
            'total_fallbacks': int(self.metrics['fallback_triggers']),
            'avg_episode_length': float(np.mean(self.metrics['episode_lengths']) if self.metrics['episode_lengths'] else 0.0),
            'collision_rate': float((self.metrics['collisions'] / n_episodes) * 100),
            'avg_lane_deviation': float(np.mean(self.metrics['lane_deviation_log']) if self.metrics['lane_deviation_log'] else 0.0),
            'average_reward': float(avg_reward),
            'average_uncertainty_rl_active': float(avg_uncertainty_rl_active), # This is C when RL is active
            'fallback_activation_rate_percent': float(fallback_activation_rate_percent) # Overall fallback rate
        }
        
        print("\nEvaluation Summary:")
        for k, v in summary.items():
            print(f"{k.replace('_', ' ').title()}: {v:.2f}")
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(summary, f, indent=4)

        # Plot lane deviation with paper's styling
        plt.figure(figsize=(8, 4))
        plt.plot(self.metrics['lane_deviation_log'], color='#2c7bb6', linewidth=2)
        plt.title("Lane Centering Performance", fontsize=12, fontweight='bold')
        plt.xlabel("Episode Number", fontsize=10)
        plt.ylabel("Average Deviation (m)", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("plots/lane_deviation.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Plot episode rewards with paper's styling
        plt.figure(figsize=(8, 4))
        plt.plot(self.metrics['episode_rewards'], color='#d7191c', linewidth=2)
        plt.title("Learning Progress", fontsize=12, fontweight='bold')
        plt.xlabel("Training Episode", fontsize=10)
        plt.ylabel("Accumulated Reward", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig("plots/training_progress.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Model uncertainty plot 
        if self.metrics['episode_step_uncertainties']:
            plt.figure(figsize=(8, 6))
            sample_episode_idx = 0
            uncertainties = self.metrics['episode_step_uncertainties'][sample_episode_idx]
            threshold = self.ensemble_wrapper.threshold
            
            # Fix: Use range for x-axis instead of undefined 'episodes'
            time_steps = range(len(uncertainties))
            plt.plot(time_steps, uncertainties, color='blue', linewidth=1.5, label='Model Uncertainty')
            plt.axhline(y=threshold, color='#984ea3', linestyle='--', linewidth=2, 
                       label=f'Safety Threshold ({threshold:.2f})')
            plt.ylim(1, max(uncertainties) + 0.5)  # Start y-axis from 1
            plt.fill_between(time_steps, threshold, uncertainties, 
                             where=(np.array(uncertainties) > threshold), color='gray', alpha=0.3, step='mid')
            plt.title("Uncertainty Monitoring and Safety Intervention", fontsize=12, fontweight='bold')
            plt.xlabel("Time Step", fontsize=10)
            plt.ylabel("Uncertainty Measure", fontsize=10)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig("plots/uncertainty_monitoring.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Calculate activation rates per episode
        activation_rates = [np.mean(ep_fallbacks) * 100 for ep_fallbacks in self.metrics['episode_step_fallbacks']]

        # Fallback activation plot with paper's styling
        if self.metrics['episode_step_fallbacks']:
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(activation_rates)), activation_rates, 
                    color=['#ff7f00' if rate > 0 else '#377eb8' for rate in activation_rates])
            plt.title("Safety Intervention Frequency", fontsize=12, fontweight='bold')
            plt.xlabel("Episode Number", fontsize=10)
            plt.ylabel("Intervention Rate (%)", fontsize=10)
            plt.savefig("plots/safety_intervention.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Plot distribution of uncertainties
        plt.figure()
        if self.metrics['collected_uncertainties']:
            plt.hist(self.metrics['collected_uncertainties'], bins=50)
            plt.title("Distribution of Uncertainties (RL Active Steps)")
            plt.xlabel("Uncertainty Value")
            plt.ylabel("Frequency")
        else:
            plt.text(0.5, 0.5, "No RL agent steps to plot uncertainty", ha='center', va='center')
        plt.grid(True)
        plt.savefig("uncertainty_distribution_plot.png")
        plt.close()

        # Plot 1: Model Uncertainty & Fallback Switches Over Time 
        if self.metrics['episode_step_uncertainties'] and self.metrics['episode_step_fallbacks']:
            plt.figure(figsize=(12, 6))
            
            sample_episode_idx = 0
            uncertainties = self.metrics['episode_step_uncertainties'][sample_episode_idx]
            fallbacks = self.metrics['episode_step_fallbacks'][sample_episode_idx]
            time_steps = np.arange(len(uncertainties))

            plt.plot(time_steps, uncertainties, label='Model Uncertainty (C)', color='blue')
            plt.axhline(y=self.ensemble_wrapper.threshold, color='red', linestyle='--', label=f'Uncertainty Threshold ({self.ensemble_wrapper.threshold:.2f})')
            
            # Highlight fallback periods
            for i in range(len(fallbacks)):
                if fallbacks[i] == 1:
                    plt.fill_between([time_steps[i]-0.5, time_steps[i]+0.5], plt.ylim()[0], plt.ylim()[1], color='gray', alpha=0.3, step='mid')

            # Create a dummy patch for the fallback legend
            from matplotlib.patches import Patch
            legend_elements = [
                plt.Line2D([0], [0], color='blue', lw=2, label='Model Uncertainty (C)'),
                plt.Line2D([0], [0], color='red', linestyle='--', lw=2, label=f'Uncertainty Threshold ({self.ensemble_wrapper.threshold:.2f})'),
                Patch(facecolor='gray', alpha=0.3, label='Fallback Active')
            ]
            plt.legend(handles=legend_elements)
            plt.title(f"Model Uncertainty & Fallback Switches (Episode {sample_episode_idx + 1})")
            plt.xlabel("Time Step in Episode")
            plt.ylabel("Uncertainty Value")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("model_uncertainty_over_time_plot.png")
            plt.close()

        # Plot 2: Fallback Activation Rate Per Episode
        if self.metrics['episode_step_fallbacks']:
            plt.figure(figsize=(10, 5))
            activation_rates_per_episode = [np.mean(ep_fallbacks) * 100 for ep_fallbacks in self.metrics['episode_step_fallbacks']]
            plt.plot(np.arange(1, len(activation_rates_per_episode) + 1), activation_rates_per_episode, marker='o', linestyle='-')
            plt.title("Fallback Activation Rate Per Episode")
            plt.xlabel("Episode Number")
            plt.ylabel("Fallback Activation Rate (%)")
            plt.ylim(0, 100)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("fallback_activation_rate_per_episode_plot.png")
            plt.close()

        # Create average speed plot
        plt.figure(figsize=(8, 6))
        if len(self.metrics['average_speed']) > 1:
            episodes = range(1, len(self.metrics['average_speed']) + 1)
            speeds = self.metrics['average_speed']
            
            plt.plot(episodes, speeds, color='green', linewidth=1.5, label='Average Speed')
            plt.axhline(y=25.36, color='red', linestyle='--', label='Target Speed (25.36 m/s)')
            plt.ylim(0, 80)  # Set speed y-axis to 80
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Average Speed (m/s)', fontsize=12)
            plt.title('Average Speed during Evaluation')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig("plots/average_speed.png", dpi=300)
            plt.close()

        # Create uncertainty over episodes plot
        plt.figure(figsize=(8, 6))
        # If we have uncertainty data for multiple episodes
        if len(self.metrics['episode_step_uncertainties']) > 1:
            # Calculate average uncertainty per episode
            avg_uncertainties = [np.mean(ep_uncertainties) for ep_uncertainties in self.metrics['episode_step_uncertainties']]
            episodes = range(1, len(avg_uncertainties) + 1)
            
            plt.plot(episodes, avg_uncertainties, color='blue', linewidth=1.5)
            plt.xlabel('Episodes', fontsize=12)
            plt.ylabel('Uncertainty', fontsize=12)
            plt.title('Average Uncertainty Value Curve during Evaluation')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig("plots/uncertainty_over_episodes.png", dpi=300)
            plt.close()
        
        # Create reward convergence plot
        plt.figure(figsize=(8, 6))
        if len(self.metrics['episode_rewards']) > 1:
            episodes = range(1, len(self.metrics['episode_rewards']) + 1)
            rewards = self.metrics['episode_rewards']
            
            plt.plot(episodes, rewards, color='blue', linewidth=1.5, label='Rewards')
            plt.ylim(0, 500)  # Set reward y-axis to 500
            
            # Add confidence interval if we have enough data points
            if len(rewards) > 5:
                window_size = min(5, len(rewards) // 2)
                rewards_smooth = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                episodes_smooth = range(window_size, len(rewards) + 1)
                
                # Calculate a simple confidence interval
                std_dev = np.std(rewards) * 0.5
                plt.fill_between(
                    episodes_smooth, 
                    rewards_smooth - std_dev, 
                    rewards_smooth + std_dev, 
                    alpha=0.3, 
                    color='lightblue'
                )
            
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Rewards', fontsize=12)
            plt.title('Reward Convergence during Evaluation')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig("plots/reward_convergence.png", dpi=300)
            plt.close()


if __name__ == "__main__":
    evaluator = Evaluator("sac_highway_model")
    evaluator.evaluate(n_episodes=3500)
