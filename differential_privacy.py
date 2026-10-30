# differential_privacy.py - CORRECTED VERSION
import numpy as np
import math
from scipy import optimize

class DifferentialPrivacy:
    def __init__(self, epsilon, delta, R_max, theta=None):
        self.epsilon = epsilon
        self.delta = delta
        self.R_max = R_max
        self.theta = theta
        
    def compute_sensitivity(self, privacy_type='per_rating'):
        R_32 = self.R_max ** 1.5  # R^{3/2}
        
        if privacy_type == 'per_rating':
            return 2 * R_32
            
        elif privacy_type == 'per_user':
            if self.theta is None:
                raise ValueError("Theta required for per-user privacy")
            return 2 * math.sqrt(2) * self.theta * R_32
            
        else:
            raise ValueError(f"Unknown privacy type: {privacy_type}")
    
    def gaussian_noise(self, shape, sensitivity, noise_multiplier=1.0):
        sigma = sensitivity * np.sqrt(noise_multiplier)
        return np.random.normal(0, sigma, shape)
    
    def apply_random_trimming(self, ratings, theta):
        if not ratings or len(ratings) == 0:
            return ratings
            
        user_ratings = {}
        for user, item, rating in ratings:
            if user not in user_ratings:
                user_ratings[user] = []
            user_ratings[user].append((user, item, rating))
        
        trimmed = []
        for user, urs in user_ratings.items():
            if len(urs) > theta:
                indices = np.random.choice(len(urs), theta, replace=False)
                trimmed.extend([urs[i] for i in indices])
            else:
                trimmed.extend(urs)
                
        return trimmed
    
    def _compute_log_moment(self, q, sigma, lambd):
        if q == 0 or sigma == 0:
            return float('inf')
            
        if lambd <= sigma**2 * np.log(1/(q*sigma + 1e-10)):
            return lambd * (lambd + 1) / (2 * sigma**2)
        else:
            return 0.5 * lambd**2 / sigma**2
    
    def compute_epsilon(self, sigma, q, steps, target_delta=None):
        if target_delta is None:
            target_delta = self.delta
            
        best_epsilon = float('inf')
        
        for lambd in range(1, 65):
            alpha_one_step = self._compute_log_moment(q, sigma, lambd)
            
            alpha_total = steps * alpha_one_step
            
            epsilon = (alpha_total - np.log(target_delta)) / lambd
            
            best_epsilon = min(best_epsilon, epsilon)
            
        return best_epsilon
    
    def compute_noise_multiplier(self, target_epsilon, q, steps, target_delta=None):
        if target_delta is None:
            target_delta = self.delta
            
        
        sigma_low, sigma_high = 0.1, 100.0
        tolerance = 1e-4
        
        for _ in range(50): 
            sigma_mid = (sigma_low + sigma_high) / 2
            epsilon_mid = self.compute_epsilon(sigma_mid, q, steps, target_delta)
            
            if epsilon_mid < target_epsilon:
                sigma_high = sigma_mid
            else:
                sigma_low = sigma_mid
                
            if sigma_high - sigma_low < tolerance:
                break
                
        return sigma_high
    
    def compute_noise_multiplier_proposition2(self, T, T_prime, sampling_rate, c1=2.0):
        if self.epsilon <= 0 or self.delta <= 0:
            return 0.0
            
        total_steps = T * T_prime
        
        numerator = c1 * (sampling_rate**2) * total_steps * math.log(1.0 / self.delta)
        denominator = (self.epsilon**2)
        
        if denominator <= 0:
            return 0.0
            
        z_squared = numerator / denominator
        z = math.sqrt(z_squared)
        
        return z
    
    def compute_epsilon_proposition2(self, T, T_prime, sampling_rate, noise_multiplier, c1=2.0):

        if noise_multiplier <= 0:
            return float('inf')
            
        total_steps = T * T_prime
        
        # Rearrange formula: ε² = c₁ * (η² * T * T′ * ln(1/δ)) / z²
        numerator = c1 * (sampling_rate**2) * total_steps * math.log(1.0 / self.delta)
        denominator = (noise_multiplier**2)
        
        if denominator <= 0:
            return float('inf')
            
        epsilon_squared = numerator / denominator
        epsilon = math.sqrt(epsilon_squared)
        
        return epsilon
    
    def get_privacy_spent(self, sigma, q, steps, target_delta=None):
        if target_delta is None:
            target_delta = self.delta
            
        epsilon = self.compute_epsilon(sigma, q, steps, target_delta)
        
        return {
            'epsilon': epsilon,
            'delta': target_delta,
            'sigma': sigma,
            'sampling_rate': q,
            'steps': steps,
            'privacy_type': 'Rényi DP' if steps > 1 else 'Pure DP'
        }
    
    def get_privacy_spent_proposition2(self, T, T_prime, sampling_rate, noise_multiplier, c1=0.1):
        epsilon = self.compute_epsilon_proposition2(T, T_prime, sampling_rate, noise_multiplier, c1)
        
        return {
            'epsilon': epsilon,
            'delta': self.delta,
            'noise_multiplier': noise_multiplier,
            'sampling_rate': sampling_rate,
            'T': T,
            'T_prime': T_prime,
            'total_steps': T * T_prime,
            'privacy_type': 'Proposition 2 (VFL/HFL)'
        }