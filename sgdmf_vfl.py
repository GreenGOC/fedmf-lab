import numpy as np
import logging

from differential_privacy import DifferentialPrivacy
from coordinator import Coordinator
from party import Party

logger = logging.getLogger(__name__)

class VFL_SGDMF:
    def __init__(self, n_parties, n_users, n_items, embedding_dim, R_max, epsilon, delta, privacy_type='per_rating', theta=10, party_data_list=None):
        self.setting = 'VFL'
        self.n_parties = n_parties
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.R_max = R_max
        self.dp = DifferentialPrivacy(epsilon, delta, R_max, theta)
        self.theta = theta
        self.privacy_type = privacy_type
        
        self.coordinator = Coordinator('VFL', n_users, n_items, embedding_dim, R_max)
        self.parties = []
        self.loss_history = []
        self.privacy_spent = []
        
        self.party_items = []
        self.party_V = []
        self.global_U = None
        
        if party_data_list is not None:
            self.initialize(party_data_list)
    
    def initialize(self, party_data_list):
        scale = np.sqrt(self.R_max / self.embedding_dim)
        self.global_U = np.random.rand(self.n_users, self.embedding_dim) * scale
        self.global_U = self._clip_embeddings(self.global_U)
        
        for k in range(self.n_parties):
            items = list(set([item_index for _, item_index, _ in party_data_list[k]]))
            self.party_items.append(items)
            
            V_k = np.random.rand(len(items), self.embedding_dim) * scale
            V_k = self._clip_embeddings(V_k)
            self.party_V.append(V_k)
            
            party_data_list[k] = [tuple(row) for row in party_data_list[k]]
            party = Party(k, party_data_list[k], 'VFL', self.embedding_dim, self.R_max)
            self.parties.append(party)
            self.coordinator.add_party(party)
    
    def _clip_embeddings(self, embeddings):
        embeddings = np.maximum(embeddings, 0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        scaling = np.minimum(1, np.sqrt(self.R_max) / (norms + 1e-8))
        return embeddings * scaling
    
    def train(self, party_data_list, T, T_prime, learning_rate, sampling_rate=0.01, add_noise=True):
        if self.privacy_type == 'per_user':
            trimmed_data = []
            for k in range(self.n_parties):
                trimmed = self.dp.apply_random_trimming(party_data_list[k], self.theta)
                trimmed_data.append(trimmed)
                self.parties[k].local_data = trimmed
            party_data_list = trimmed_data
        
        sensitivity = self.dp.compute_sensitivity(self.privacy_type)
        total_steps = T * T_prime 
        
        noise_multiplier = self.dp.compute_noise_multiplier(
            self.dp.epsilon, sampling_rate, total_steps, self.dp.delta
        )
        
        if noise_multiplier is None:
            logger.warning("Failed to compute noise multiplier, disabling noise")
            add_noise = False
        elif noise_multiplier > 100:  # Sanity check
            logger.warning(f"Noise multiplier too high ({noise_multiplier}), reducing noise")
            noise_multiplier = min(noise_multiplier, 10.0)

        for t in range(T):
            party_U_updates = []
            total_loss = 0
            
            for k in range(self.n_parties):
                party = self.parties[k]
                U_k = self.global_U.copy()
                V_k = self.party_V[k].copy()
                
                party.receive_global_embeddings(U_k)
                party.local_embeddings = V_k
                
                local_loss = 0
                for _ in range(T_prime):
                    if self.privacy_type == 'per_user':
                        batch = party._sample_by_rate(sampling_rate, 'user')
                    else:
                        batch = party._sample_by_rate(sampling_rate, 'rating')
                    
                    if len(batch) == 0:
                        continue
                    
                    grad_U, grad_V = party._compute_gradients_vfl(batch, U_k, V_k)
                    
                    if len(batch) > 0:
                        grad_U = grad_U / len(batch)
                        grad_V = grad_V / len(batch)
                    
                    if add_noise and noise_multiplier is not None:
                        noise_U = self.dp.gaussian_noise(grad_U.shape, sensitivity, noise_multiplier)
                        noise_V = self.dp.gaussian_noise(grad_V.shape, sensitivity, noise_multiplier)
                        
                        grad_U += noise_U
                        grad_V += noise_V
                    
                    U_k -= learning_rate * grad_U
                    V_k -= learning_rate * grad_V
                    
                    U_k = self._clip_embeddings(U_k)
                    V_k = self._clip_embeddings(V_k)
                    
                    batch_loss = party._compute_loss_vfl(batch, U_k, V_k)
                    local_loss += batch_loss
                
                party_U_updates.append(U_k)
                self.party_V[k] = V_k
                self.parties[k].local_embeddings = V_k
                
                total_loss += local_loss / T_prime if T_prime > 0 else 0
            
            weights = []
            for k in range(self.n_parties):
                if len(party_data_list[k]) > 0:
                    weights.append(len(party_data_list[k]))
                else:
                    weights.append(1)  
            
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / self.n_parties] * self.n_parties
            
            aggregated_U = np.zeros_like(self.global_U)
            for w, U_k in zip(weights, party_U_updates):
                aggregated_U += w * U_k
            
            self.global_U = aggregated_U
            self.coordinator.global_embeddings = self.global_U
            
            for party in self.parties:
                party.receive_global_embeddings(self.global_U)
            
            avg_loss = total_loss / self.n_parties
            self.loss_history.append(avg_loss)
            logger.info(f"Round {t+1}/{T}, Loss: {avg_loss:.4f}, LR: {learning_rate:.6f}")
            
            for k in range(self.n_parties):
                party = self.parties[k]
                party.fine_tune(5, learning_rate * 0.1)  
                self.party_V[k] = party.local_embeddings
            
            if t / T <= 0.8 and t > 0:
                learning_rate *= 0.98
        
        logger.info("Starting final fine-tuning phase...")
        for k in range(self.n_parties):
            party = self.parties[k]
            party.fine_tune(20, learning_rate * 0.05)
            self.party_V[k] = party.local_embeddings
        
        return self.loss_history
    
    def predict(self, user_index, item_index):
        for k in range(self.n_parties):
            if item_index in self.party_items[k]:
                local_idx = self.party_items[k].index(item_index)
                return np.dot(self.global_U[user_index], self.party_V[k][local_idx])
        return 0.0