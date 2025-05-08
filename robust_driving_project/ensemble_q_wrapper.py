import numpy as np
import torch

class EnsembleWrapper:
    def __init__(self, policy, uncertainty_threshold=0.2):
        self.policy = policy
        self.threshold = uncertainty_threshold

    def predict_q_values(self, obs):
        # Convert observation to tensor and handle batch dimension
        if len(obs.shape) == 3:
            obs = obs[0]
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        # Get actions from policy
        with torch.no_grad():
            # SAC's actor returns a tuple (actions, log_prob)
            action = self.policy.actor(obs_tensor)
            if isinstance(action, tuple):
                action = action[0]
            
            # Concatenate observation and action for critic input
            critic_input = torch.cat([obs_tensor, action], dim=1)
            
            # Get Q-values from both critics
            q1 = self.policy.critic.qf0(critic_input)
            q2 = self.policy.critic.qf1(critic_input)
            
            # Convert to numpy and remove batch dimension
            q1 = q1.detach().numpy().squeeze()
            q2 = q2.detach().numpy().squeeze()
        
        # Create ensemble predictions
        q_values = [q1, q2]
        
        # Add noisy predictions to simulate larger ensemble
        for _ in range(3):
            noise = np.random.normal(0, 0.05, q1.shape)
            noisy_q = (q1 + q2) / 2 + noise
            q_values.append(noisy_q)
        
        return q_values

    def compute_uncertainty(self, q_values):
        # Compute standard deviation across ensemble predictions
        q_values = np.array(q_values)
        return np.std(q_values, axis=0).mean()

    def should_use_fallback(self, obs):
        try:
            q_values = self.predict_q_values(obs)
            uncertainty = self.compute_uncertainty(q_values)
            return uncertainty > self.threshold, uncertainty
        except Exception as e:
            print(f"[EnsembleWrapper Error]: {e}")
            return True, 1.0  # default to fallback
