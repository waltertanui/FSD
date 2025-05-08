import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
from stable_baselines3.sac.policies import SACPolicy

class EnsembleQNetworks:
    def __init__(self, state_dim: int, action_dim: int, ensemble_size: int = 5):
        self.ensemble_size = ensemble_size
        self.networks = []
        
        for _ in range(ensemble_size):
            net = nn.Sequential(
                nn.Linear(state_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            self.networks.append(net)
    
    def get_uncertainty(self, state: np.ndarray, action: np.ndarray) -> float:
        q_values = []
        
        # Handle vectorized environment observations (remove batch dimension if present)
        if len(state.shape) == 3:
            state = state[0]  # Take the first (and only) environment
        if len(action.shape) == 2:
            action = action[0]  # Take the first (and only) action
        
        # Ensure both tensors have the same number of dimensions
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.FloatTensor(action)
        
        # If dimensions don't match, reshape action to match state dimensions
        if len(state_tensor.shape) != len(action_tensor.shape):
            if len(state_tensor.shape) > len(action_tensor.shape):
                action_tensor = action_tensor.view(1, -1)
            else:
                state_tensor = state_tensor.view(1, -1)
        
        state_action = torch.cat([state_tensor, action_tensor], dim=-1)
        
        for net in self.networks:
            q_values.append(net(state_action).detach().numpy())
            
        return np.var(q_values)

class EnsembleWrapper:
    def __init__(self, policy: SACPolicy, uncertainty_threshold: float = 0.5):
        self.policy = policy
        self.uncertainty_threshold = uncertainty_threshold
        self.ensemble = EnsembleQNetworks(
            state_dim=policy.observation_space.shape[0],
            action_dim=policy.action_space.shape[0]
        )
    
    def should_use_fallback(self, state: np.ndarray) -> Tuple[bool, float]:
        action = self.policy.predict(state, deterministic=True)[0]
        uncertainty = self.ensemble.get_uncertainty(state, action)
        return uncertainty > self.uncertainty_threshold, uncertainty