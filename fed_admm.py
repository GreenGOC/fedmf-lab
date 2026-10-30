import numpy as np
import logging
from party import Party
from coordinator import Coordinator

logger = logging.getLogger(__name__)

class FedADMM:    
    def __init__(self, n_parties, n_users, n_items, embedding_dim, R_max, setting='HFL', rho=1.0):
        self.setting = setting
        self.n_parties = n_parties
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.R_max = R_max
        self.rho = rho  
        
        self.coordinator = Coordinator(setting, n_users, n_items, embedding_dim, R_max)
        self.parties = []
        self.party_users = []
        self.lagrange_multipliers = []  
        self.local_copies = []  
        
        self.loss_history = []  # Track global loss over rounds
    
    def initialize_parties(self, party_data_list):
        for k in range(self.n_parties):
            users = list(set([user_index for user_index, _, _ in party_data_list[k]]))
            self.party_users.append(users)
            
            party = Party(k, party_data_list[k], self.setting, self.embedding_dim, self.R_max)
            self.coordinator.add_party(party)
            self.parties.append(party)
            
            if self.setting == 'VFL':
                shape = (self.n_users, self.embedding_dim)
            else:
                shape = (self.n_items, self.embedding_dim)
            
            self.lagrange_multipliers.append(np.zeros(shape))
            self.local_copies.append(self.coordinator.global_embeddings.copy())
            
    def predict(self, user_index, item_index):
        for k in range(self.n_parties):
            if user_index in self.party_users[k]:
                party = self.parties[k]
                return party.predict(user_index, item_index)
        return 0.0
            
    def train(self, party_data_list, T, local_epochs, learning_rate, sample_fraction=1.0):
        self.coordinator.broadcast_embeddings()
        for t in range(T):
            party_updates = []
            party_losses = []
            
            for i, party in enumerate(self.parties):
                party_index = party.party_id
                
                if self.setting == 'VFL':
                    update, loss = self._local_update_vfl(
                        party, party_data_list[party_index],
                        self.local_copies[i], self.lagrange_multipliers[i],
                        local_epochs, learning_rate
                    )
                else:
                    update, loss = self._local_update_hfl(
                        party, party_data_list[party_index],
                        self.local_copies[i], self.lagrange_multipliers[i],
                        local_epochs, learning_rate
                    )
                
                party_updates.append(update)
                party_losses.append(loss)
                logger.debug(f"Party {party_index}: loss = {loss:.4f}")
            
            aggregated = self._global_update(party_updates)
            self.coordinator.global_embeddings = aggregated
            
            for i in range(self.n_parties):
                self.local_copies[i] = self.coordinator.global_embeddings.copy()
                
                diff = self.parties[i].global_embeddings - self.local_copies[i]
                
                self.lagrange_multipliers[i] += self.rho * diff
            
            self.coordinator.broadcast_embeddings()
            
            global_loss = self._compute_global_loss(party_data_list)
            self.loss_history.append(global_loss)
            logger.info(f"Round {t+1}: Global loss = {global_loss:.4f}")
            if(t / T <= 0.8):
                learning_rate *= 0.98
                
    def _local_update_vfl(self, party, local_data, z, lambda_mult, local_epochs, learning_rate):
        U = party.global_embeddings.copy()
        V = party.local_embeddings.copy()
        
        for _ in range(local_epochs):
            batch = party._sample_batch(32)
            
            grad_U, grad_V = party._compute_gradients_vfl(batch, U, V)
            
            grad_U += self.rho * (U - z) + lambda_mult
            
            U -= learning_rate * grad_U
            V -= learning_rate * grad_V
            
            U = party._clip_embeddings(U)
            V = party._clip_embeddings(V)
        
        party.global_embeddings = U
        party.local_embeddings = V
        
        loss = party._compute_loss_vfl(batch, U, V)
        loss += np.sum(lambda_mult * (U - z))
        loss += 0.5 * self.rho * np.sum((U - z) ** 2)
        
        return U, loss
    
    def _local_update_hfl(self, party, local_data, z, lambda_mult, local_epochs, learning_rate):
        U = party.local_embeddings.copy()
        V = party.global_embeddings.copy()
        
        for _ in range(local_epochs):
            batch = party._sample_batch(32) if len(local_data) > 32 else local_data
            
            grad_U, grad_V = party._compute_gradients_hfl(batch, U, V)
            
            grad_V += self.rho * (V - z) + lambda_mult
            
            U -= learning_rate * grad_U
            V -= learning_rate * grad_V
            
            U = party._clip_embeddings(U)
            V = party._clip_embeddings(V)
        
        party.local_embeddings = U
        party.global_embeddings = V
        
        loss = party._compute_loss_hfl(batch, U, V)
        loss += np.sum(lambda_mult * (V - z))
        loss += 0.5 * self.rho * np.sum((V - z) ** 2)
        
        return V, loss
    
    def _global_update(self, local_updates):
        return np.mean(local_updates, axis=0)
    
    def _compute_global_loss(self, party_data_list):
        total_loss = 0
        total_samples = 0
        
        for i, party in enumerate(self.parties):
            party_idx = party.party_id
            
            if self.setting == 'VFL':
                loss = party._compute_loss_vfl(
                    party_data_list[party_idx],
                    party.global_embeddings,
                    party.local_embeddings
                )
                loss += np.sum(self.lagrange_multipliers[i] * (party.global_embeddings - self.local_copies[i]))
                loss += 0.5 * self.rho * np.sum((party.global_embeddings - self.local_copies[i]) ** 2)
            else:
                loss = party._compute_loss_hfl(
                    party_data_list[party_idx],
                    party.local_embeddings,
                    party.global_embeddings
                )
                loss += np.sum(self.lagrange_multipliers[i] * (party.global_embeddings - self.local_copies[i]))
                loss += 0.5 * self.rho * np.sum((party.global_embeddings - self.local_copies[i]) ** 2)
            
            total_loss += loss * len(party_data_list[party_idx])
            total_samples += len(party_data_list[party_idx])
        
        return total_loss / total_samples if total_samples > 0 else 0
    
    def embedding_clip_unified(embeddings, R_max, item_embedding=False):
        embeddings = np.maximum(embeddings, 0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        if item_embedding:
            max_norm = np.sqrt(R_max)
        else:
            max_norm = np.sqrt(R_max)
        
        scaling = np.minimum(1, max_norm / (norms + 1e-8))
        return embeddings * scaling