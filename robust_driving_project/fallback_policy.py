from typing import List  # Add this import
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
        # Handle DummyVecEnv observation format
        if len(observation.shape) == 3:
            observation = observation[0]  # Get first environment's observation
        
        # Extract relevant information from observation
        ego_state = observation[0]  # First vehicle is ego vehicle
        other_vehicles = observation[1:]  # Rest are other vehicles
        
        # Calculate longitudinal action (acceleration)
        front_vehicle = self._get_front_vehicle(ego_state, other_vehicles)
        acceleration = self._compute_idm_acceleration(ego_state, front_vehicle)
        
        # Calculate lateral action (steering)
        steering = self._compute_mobil_steering(ego_state, other_vehicles)
        
        action = np.array([acceleration, steering])
        action = np.clip(action, -1.0, 1.0)
        
        return action, None
    
    def _get_front_vehicle(self, ego_state: np.ndarray, other_vehicles: List[np.ndarray]):
        # Observation features: ['presence', 'x', 'y', 'vx', 'vy']
        # Ego state structure: [presence, x, y, vx, vy]
        ego_x = ego_state[1]  # Get x coordinate (index 1)
        ego_y = ego_state[2]  # Get y coordinate (index 2)
        
        closest_vehicle = None
        min_distance = float('inf')
        
        for vehicle in other_vehicles:
            if vehicle[0] > 0:  # Check presence
                vx = vehicle[1] - ego_x
                vy = vehicle[2] - ego_y
                distance = np.sqrt(vx**2 + vy**2)
                
                if distance < min_distance and abs(vy) < 2.0:  # Same lane threshold
                    closest_vehicle = vehicle
                    min_distance = distance
                    
        return closest_vehicle
    
    def _compute_idm_acceleration(self, ego_state, front_vehicle):
        ego_speed = ego_state[4]  # vx is at index 4
        
        if front_vehicle is None:
            # No vehicle ahead, accelerate to desired speed
            return (self.desired_speed - ego_speed) / self.desired_speed
        
        # Calculate IDM acceleration
        distance = np.sqrt(np.sum((front_vehicle[:2] - ego_state[:2])**2))
        relative_speed = front_vehicle[4] - ego_speed
        
        # Add safety check for minimum distance
        distance = max(distance, 0.1)  # Prevent division by zero
        
        desired_gap = (self.safe_distance + 
                      ego_speed * self.time_headway)
        
        acceleration = np.clip(
            (1 - (ego_speed/self.desired_speed)**2 - 
             (desired_gap/distance)**2),
            -1.0, 1.0
        )
        
        return acceleration
    
    def _compute_mobil_steering(self, ego_state, other_vehicles):
        # PID controller for lane centering
        ego_y = ego_state[2]  # y-position at index 2
        lane_center = (int(ego_y // 4) * 4) + 2  # Calculate center of current lane
        error = lane_center - ego_y
        
        # PID parameters (proportional, integral, derivative)
        Kp = 0.5
        Ki = 0.01
        Kd = 0.1
        dt = 0.2  # time step
        
        # Store errors for PID calculation
        if not hasattr(self, 'prev_error'):
            self.prev_error = error
            self.integral = 0
            
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        steering = Kp * error + Ki * self.integral + Kd * derivative
        self.prev_error = error
        
        # Clip steering to reasonable values
        return np.clip(steering, -0.3, 0.3)
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