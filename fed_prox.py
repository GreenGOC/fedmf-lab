import numpy as np
import logging
from party import Party
from coordinator import Coordinator
from fed_avg import FedAvg

logger = logging.getLogger(__name__)

class FedProx(FedAvg):
    def __init__(self, n_parties, n_users, n_items, embedding_dim, R_max, setting='HFL', mu=0.01):
        super().__init__(n_parties, n_users, n_items, embedding_dim, R_max, setting)
        self.mu = mu
        self.loss_history = []  # Track global loss over rounds
        
    def train(self, party_data_list, T, local_epochs, learning_rate, sample_fraction=1.0):
        self.coordinator.broadcast_embeddings()
        
        for t in range(T):
            logger.info(f"FedProx Round {t+1}/{T}")  # Fixed: Changed from "FedAvg" to "FedProx"
            if sample_fraction < 1.0:
                num_sampled = max(1, int(self.n_parties* sample_fraction))
                sampled_indices = np.random.choice(self.n_parties, num_sampled, replace=False)
                sampled_parties = [self.parties[i] for i in sampled_indices]
            else:
                sampled_parties = self.parties
                
            party_updates =[]
            party_sizes = []
            for party in sampled_parties:
                party_index = party.party_id
                global_before = party.global_embeddings.copy()
                
                local_loss = 0
                for epoch in range(local_epochs):
                    if self.setting == 'VFL':
                        update, loss = self._train_local_with_proximal_vfl(
                            party, party_data_list[party_index], global_before,
                            learning_rate, self.mu
                        )
                    else:
                        update, loss = self._train_local_with_proximal_hfl(
                            party, party_data_list[party_index], global_before,
                            learning_rate, self.mu
                        )
                        
                    local_loss += loss
                    
                party_updates.append(update)
                party_sizes.append(len(party_data_list[party_index]))
                logger.debug(f"Party {party_index}: local loss = {local_loss/local_epochs:.4f}")
            
            total_size = sum(party_sizes)
            aggregated = self.coordinator.aggregate_embeddings(party_updates)
            self.coordinator.global_embeddings = aggregated
            
            self.coordinator.broadcast_embeddings()
            # Track loss every round
            global_loss = self._compute_global_loss(party_data_list)
            self.loss_history.append(global_loss)
            
            logger.info(f"Round {t+1}: Global loss = {global_loss:.4f}")
            if(t / T <= 0.8):
                learning_rate *= 0.98
                
    def _train_local_with_proximal_vfl(self, party, local_data, global_U, learning_rate, mu):
        U = party.global_embeddings.copy()
        V = party.local_embeddings.copy()
        
        batch = party._sample_batch(32)
        grad_U, gradient_V = party._compute_gradients_vfl(batch, U, V)
        grad_U += mu * (U - global_U)
        U -= learning_rate * grad_U
        V -= learning_rate * gradient_V
        
        U = party._clip_embeddings(U)
        V = party._clip_embeddings(V)
        
        party.global_embeddings = U
        party.local_embeddings = V
        
        loss = party._compute_loss_vfl(batch, U, V) + 0.5 * mu * np.sum((U-global_U) ** 2)
        return U, loss
    
    def _train_local_with_proximal_hfl(self, party, local_data, global_V, learning_rate, mu):
        U = party.local_embeddings.copy()
        V = party.global_embeddings.copy()
        
        batch = party._sample_batch(32) if len(local_data) > 32 else local_data
        
        grad_U, grad_V = party._compute_gradients_hfl(batch, U, V)
        
        grad_V += mu * (V - global_V)
        
        U -= learning_rate * grad_U
        V -= learning_rate * grad_V
        
        U = party._clip_embeddings(U)
        V = party._clip_embeddings(V)
        
        party.local_embeddings = U
        party.global_embeddings = V
        
        loss = party._compute_loss_hfl(batch, U, V) + 0.5 * mu * np.sum((V - global_V) ** 2)
        return V, loss
    
    def embedding_clip_unified(embeddings, R_max, item_embedding=False):
        embeddings = np.maximum(embeddings, 0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        if item_embedding:
            max_norm = np.sqrt(R_max)
        else:
            max_norm = np.sqrt(R_max)
        
        scaling = np.minimum(1, max_norm / (norms + 1e-8))
        return embeddings * scaling