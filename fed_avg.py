import numpy as np
import logging 
from party import Party
from coordinator import Coordinator

logger = logging.getLogger(__name__)

class FedAvg:
    def __init__(self, n_parties, n_users, n_items, embedding_dim, R_max, setting='HFL'):
        self.setting = setting
        self.n_parties = n_parties
        self.n_users = n_users 
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.R_max = R_max
        
        self.coordinator = Coordinator(setting, n_users, n_items, embedding_dim, R_max)
        self.parties = []
        self.loss_history = []  # Track global loss over rounds
        self.party_users = []
    
    def predict(self, user_index, item_index):
        for k in range(self.n_parties):
            if user_index in self.party_users[k]:
                party = self.parties[k]
                return party.predict(user_index, item_index)
        return 0.0
    
    def initialize_parties(self, party_data_list):
        for k in range(self.n_parties):
            party = Party(k,  party_data_list[k], self.setting, self.embedding_dim, self.R_max)
            self.coordinator.add_party(party)
            self.parties.append(party)
            users = list(set([user_index for user_index, _, _ in party_data_list[k]]))
            self.party_users.append(users)            
    def train(self, party_data_list, T, local_epochs, learning_rate, sample_fraction=1.0):
        self.coordinator.broadcast_embeddings()
        
        for t in range(T):
            logger.info(f"FedAvg Round {t+1}/{T}")
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
                local_loss = 0
                for epoch in range(local_epochs):
                    if(self.setting == 'VFL'):
                        update, loss = party.train_local(T_prime=10, learning_rate=learning_rate, dp=None, sampling_rate=None)
                    else:
                        update, loss = party.train_local(T_prime=10, learning_rate=learning_rate, dp=None, sampling_rate=None)
                        
                    local_loss += loss
                    
                party_updates.append(update)
                party_sizes.append(len(party_data_list[party_index]))
            
            aggregated = self.coordinator.aggregate_embeddings(party_updates)
            self.coordinator.global_embeddings = aggregated
            
            self.coordinator.broadcast_embeddings()
            global_loss = self._compute_global_loss(party_data_list)
            self.loss_history.append(global_loss)
            logger.info(f"Round {t+1}: Global loss = {global_loss:.4f}")
            if(t / T <= 0.8):
                learning_rate *= 0.98
                
    def _compute_global_loss(self, party_data_list):
        total_loss=0
        total_samples=0
        
        for party in self.parties:
            party_index = party.party_id
            if(self.setting == 'VFL'):
                loss = party._compute_loss_vfl(party_data_list[party_index], party.global_embeddings, party.local_embeddings)
            else:
                loss = party._compute_loss_hfl(party_data_list[party_index], party.local_embeddings, party.global_embeddings)
                
            total_loss += loss * len(party_data_list[party_index])
            total_samples += len(party_data_list[party_index])
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


