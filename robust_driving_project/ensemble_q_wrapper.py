import numpy as np
import torch

class EnsembleWrapper:
    def __init__(self, policy, uncertainty_threshold=0.2):
        self.policy = policy
        self.threshold = uncertainty_threshold

    def predict_q_values(self, obs):
        # Handle nested dimensions from DummyVecEnv
        if len(obs.shape) == 3:
            obs = obs.reshape(obs.shape[0], -1)  # Flatten to 2D
        
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(obs)
        
        # Get actions from policy
        with torch.no_grad():
            actions = self.policy.actor(obs_tensor) # Correctly assign single output
            
            # Concatenate observations and actions for critic input
            critic_input = torch.cat([obs_tensor, actions], dim=1)
            
            # Get Q-values from both critics
            q1 = self.policy.critic.q_networks[0](critic_input)
            q2 = self.policy.critic.q_networks[1](critic_input)
            
            # Convert to numpy
            q1 = q1.detach().numpy()
            q2 = q2.detach().numpy()
        
        # Create ensemble predictions with noise
        q_values = [q1, q2]
        for _ in range(3):
            noise = np.random.normal(0, 0.05, q1.shape)
            q_values.append((q1 + q2) / 2 + noise)
        
        return np.concatenate(q_values)

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
