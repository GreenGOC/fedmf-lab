import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging
from differential_privacy import DifferentialPrivacy
from coordinator import Coordinator
from party import Party

logger = logging.getLogger(__name__)

class HFL_SGDMF:
    def __init__(self, n_parties, n_users, n_items, embedding_dim, R_max, epsilon, delta, 
                 privacy_type='per_rating', theta=10, T=100, T_prime=10):
        self.setting = 'HFL'
        self.n_parties = n_parties
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.R_max = R_max
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_type = privacy_type
        self.theta = theta
        self.T = T
        self.T_prime = T_prime
        
        self.dp = DifferentialPrivacy(epsilon, delta, R_max, theta)
        self.coordinator = Coordinator('HFL', n_users, n_items, embedding_dim, R_max)
        self.parties = []
        self.loss_history = []
        
        self.global_V = None
        self.party_users = []
        
    def initialize(self, party_data_list, pretrain_epochs=10, pretrain_lr=0.01):
        scale = np.sqrt(self.R_max / self.embedding_dim)
        self.global_V = np.random.rand(self.n_items, self.embedding_dim) * scale
        self.global_V = self._clip_embeddings(self.global_V)
        
        for k in range(self.n_parties):
            users = list(set([user_index for user_index, _, _ in party_data_list[k]]))
            self.party_users.append(users)
            
            party = Party(k, party_data_list[k], 'HFL', self.embedding_dim, self.R_max)
            self.parties.append(party)
            self.coordinator.add_party(party)
            
            party.local_embeddings = np.random.rand(len(users), self.embedding_dim) * scale
            party.local_embeddings = party._clip_embeddings(party.local_embeddings)
            
            if len(party_data_list[k]) > 0:
                self._pretrain_user_embeddings(party, party_data_list[k], pretrain_epochs, pretrain_lr)
    
    def _clip_embeddings(self, embeddings):
        embeddings = np.maximum(embeddings, 0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        scaling = np.minimum(1, np.sqrt(self.R_max) / (norms + 1e-8))
        return embeddings * scaling
    
    def _pretrain_user_embeddings(self, party, local_data, epochs, learning_rate):
        U = party.local_embeddings.copy()
        V = self.global_V.copy()
        
        party.receive_global_embeddings(V)
        
        for epoch in range(epochs):
            if len(local_data) == 0:
                continue
                
            batch_size = min(32, len(local_data))
            batch = party._sample_batch(batch_size)
            
            grad_U, _ = party._compute_gradients_hfl(batch, U, V)
            
            if len(batch) > 0:
                grad_U = grad_U / len(batch)
            
            U -= learning_rate * grad_U
            U = party._clip_embeddings(U)
            
            if epoch % 5 == 0:
                loss = party._compute_loss_hfl(batch, U, V)
                logger.debug(f"Pretraining epoch {epoch}, loss: {loss:.4f}")
        
        party.local_embeddings = U
    
    def train(self, party_data_list, T, T_prime, learning_rate, sampling_rate=0.01):
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
            self.epsilon, sampling_rate, total_steps, self.delta
        )

        for t in range(T):
            party_V_updates = []
            total_loss = 0
            
            for k in range(self.n_parties):
                party = self.parties[k]
                
                V_k = self.global_V.copy()
                party.receive_global_embeddings(V_k)
                
                local_loss = 0
                for _ in range(T_prime):
                    if self.privacy_type == 'per_user':
                        batch = party._sample_by_rate(sampling_rate, 'user')
                    else:
                        batch = party._sample_by_rate(sampling_rate, 'rating')
                    
                    if len(batch) == 0:
                        continue
                    
                    _, grad_V = party._compute_gradients_hfl(batch, party.local_embeddings, V_k)
                    
                    if len(batch) > 0:
                        grad_V = grad_V / len(batch)
                    
                    if noise_multiplier is not None and noise_multiplier > 0:
                        noise = self.dp.gaussian_noise(grad_V.shape, sensitivity, noise_multiplier)
                        grad_V += noise
                    
                    V_k -= learning_rate * grad_V
                    V_k = self._clip_embeddings(V_k)
                    
                    batch_loss = party._compute_loss_hfl(batch, party.local_embeddings, V_k)
                    local_loss += batch_loss
                
                party_V_updates.append(V_k)
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
            
            aggregated_V = np.zeros_like(self.global_V)
            for w, V_k in zip(weights, party_V_updates):
                aggregated_V += w * V_k
            
            self.global_V = aggregated_V
            self.coordinator.global_embeddings = self.global_V
            
            for party in self.parties:
                party.receive_global_embeddings(self.global_V)
            
            avg_loss = total_loss / self.n_parties
            self.loss_history.append(avg_loss)
            logger.info(f"Round {t+1}/{T}, Loss: {avg_loss:.4f}, LR: {learning_rate:.6f}")
            
            if t / T <= 0.8 and t > 0:
                learning_rate *= 0.98 
        
        logger.info("Starting fine-tuning phase...")
        kappa = 50  
        for k in range(self.n_parties):
            party = self.parties[k]
            party.local_data = party_data_list[k]
            party.fine_tune(kappa, learning_rate * 0.1) 
        
        return self.loss_history
    
    def predict(self, user_index, item_index):
        for k in range(self.n_parties):
            if user_index in self.party_users[k]:
                party = self.parties[k]
                return party.predict(user_index, item_index)
        return 0.0