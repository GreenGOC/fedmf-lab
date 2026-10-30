
from typing import Any


import numpy as np
from party import Party

class Coordinator:
    def __init__(self, setting, n_users, n_items, embedding_dim, R_max):
        self.setting = setting
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.R_max = R_max
        
        if(setting == 'VFL'):
            self.global_embeddings = np.random.rand(n_users, embedding_dim)
        else:
            self.global_embeddings = np.random.rand(n_items, embedding_dim)
        self.global_embeddings = self._clip_embeddings(self.global_embeddings)
        
        self.parties = []
        self.communication_rounds = 0
        
    def _clip_embeddings(self, embeddings):
        embeddings = np.maximum(embeddings, 0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        scaling = np.minimum(1, np.sqrt(self.R_max) / (norms + 1e-8))
        return embeddings * scaling
    
    def add_party(self, party):
        self.parties.append(party)
    
    def aggregate_embeddings(self, party_updates):
        if not party_updates:
            return self.global_embeddings
        
        aggregated = np.zeros_like(self.global_embeddings)
        if(self.setting == 'VFL'):
            sizes = [len(getattr(p, 'local_items', getattr(p, 'unique_items', []))) for p in self.parties]
        else:
            sizes = [len(getattr(p, 'local_users', getattr(p, 'unique_users', []))) for p in self.parties]

        total = sum(sizes) if sum(sizes) > 0 else len(party_updates)
        weights = [s / total for s in sizes]

        for update, weight in zip(party_updates, weights):
            aggregated += weight * update
            
        self.communication_rounds += 1
        return aggregated
    
    def broadcast_embeddings(self):
        for party in self.parties:
            party.receive_global_embeddings(self.global_embeddings)
    
    def get_communication_stats(self):
        if(self.setting == 'VFL'):
            bytes_per_round = self.n_users * self.embedding_dim * 8
        else:
            bytes_per_round = self.n_items * self.embedding_dim * 8

        return {
            'rounds': self.communication_rounds,
            'total_bytes': self.communication_rounds * bytes_per_round,
            'bytes_per_round': bytes_per_round
        }