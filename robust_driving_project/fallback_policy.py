from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
import numpy as np

class FallbackPolicy:
    def __init__(self):
        self.idm_vehicle = IDMVehicle
        self.desired_speed = 30  # m/s
        self.safe_distance = 4.0  # meters
        self.time_headway = 1.5   # seconds
        
    def predict(self, observation: np.ndarray):
        # Extract relevant information from observation
        ego_state = observation[:5]  # position, heading, speed
        other_vehicles = observation[5:].reshape(-1, 5)  # other vehicles' states
        
        # Calculate longitudinal action (acceleration)
        front_vehicle = self._get_front_vehicle(ego_state, other_vehicles)
        acceleration = self._compute_idm_acceleration(ego_state, front_vehicle)
        
        # Calculate lateral action (steering)
        steering = self._compute_mobil_steering(ego_state, other_vehicles)
        
        action = np.array([acceleration, steering])
        action = np.clip(action, -1.0, 1.0)
        
        return action, None
    
    def _get_front_vehicle(self, ego_state, other_vehicles):
        # Find the nearest vehicle ahead in the same lane
        ego_x, ego_y = ego_state[0:2]
        front_vehicle = None
        min_distance = float('inf')
        
        for vehicle in other_vehicles:
            x, y = vehicle[0:2]
            if x > ego_x:  # vehicle is ahead
                distance = np.sqrt((x - ego_x)**2 + (y - ego_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    front_vehicle = vehicle
        
        return front_vehicle
    
    def _compute_idm_acceleration(self, ego_state, front_vehicle):
        ego_speed = ego_state[4]
        
        if front_vehicle is None:
            # No vehicle ahead, accelerate to desired speed
            return (self.desired_speed - ego_speed) / self.desired_speed
        
        # Calculate IDM acceleration
        distance = np.sqrt(np.sum((front_vehicle[:2] - ego_state[:2])**2))
        relative_speed = front_vehicle[4] - ego_speed
        
        desired_gap = (self.safe_distance + 
                      ego_speed * self.time_headway)
        
        acceleration = np.clip(
            (1 - (ego_speed/self.desired_speed)**2 - 
             (desired_gap/distance)**2),
            -1.0, 1.0
        )
        
        return acceleration
    
    def _compute_mobil_steering(self, ego_state, other_vehicles):
        # Simple MOBIL-inspired lane changing
        # Returns small steering values for lane changes
        ego_y = ego_state[1]
        
        # Check if lane change is beneficial
        if self._is_lane_change_safe(ego_state, other_vehicles):
            if ego_y > 0:  # Too far right, steer left
                return -0.2
            else:  # Too far left, steer right
                return 0.2
        
        return 0.0
    
    def _is_lane_change_safe(self, ego_state, other_vehicles):
        # Simple safety check for lane changes
        ego_x, ego_y = ego_state[0:2]
        
        for vehicle in other_vehicles:
            x, y = vehicle[0:2]
            distance = np.sqrt((x - ego_x)**2 + (y - ego_y)**2)
            if distance < self.safe_distance * 2:
                return False
        
        return True
    
    def reset(self):
        pass