
import numpy as np
import logging
from differential_privacy import DifferentialPrivacy
from coordinator import Coordinator
from party import Party

logger = logging.getLogger(__name__)

class LFL_SGDMF:
    def __init__(self, n_users, n_items, embedding_dim, R_max, epsilon, delta, 
                 privacy_type='per_rating', theta=10, omega=0.3, T=10):
        self.setting = 'LFL'
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.R_max = R_max
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_type = privacy_type
        self.theta = theta
        self.omega = omega
        self.T = T
        
        self.dp = DifferentialPrivacy(epsilon, delta, R_max, theta)
        self.coordinator = Coordinator('HFL', n_users, n_items, embedding_dim, R_max)
        self.parties = []
        self.loss_history = []
        
        self.global_V = None
        
    def initialize(self, party_data_list, pretrain_epochs=5, pretrain_lr=0.01):
        self.global_V = np.random.rand(self.n_items, self.embedding_dim)
        self.global_V = self._clip_embeddings(self.global_V)
        
        for i in range(self.n_users):
            party = Party(i, party_data_list[i], 'HFL', self.embedding_dim, self.R_max)
            self.parties.append(party)
            self.coordinator.add_party(party)
            
            if len(party_data_list[i]) > 0:
                self._pretrain_user_embeddings(party, party_data_list[i], pretrain_epochs, pretrain_lr)
    
    def _clip_embeddings(self, embeddings):
        embeddings = np.maximum(embeddings, 0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        scaling = np.minimum(1, np.sqrt(self.R_max) / (norms + 1e-8))
        return embeddings * scaling
    
    def _pretrain_user_embeddings(self, party, local_data, epochs, learning_rate):
        U = party.local_embeddings.copy()
        V = self.global_V.copy()
        
        party.receive_global_embeddings(V)
        
        for _ in range(epochs):
            if len(local_data) == 0:
                continue
            batch = party._sample_batch(1)
            grad_U, _ = party._compute_gradients_hfl(batch, U, V)
            U -= learning_rate * grad_U
            U = party._clip_embeddings(U)
        
        party.local_embeddings = U
    
    def _secure_aggregation_round1(self, active_indices):
        noisy_updates = []
        survivors = []
        
        for i in active_indices:
            party = self.parties[i]
            if len(party.local_data) == 0:
                continue
            
            batch = party._sample_batch(1)
            _, grad_V = party._compute_gradients_hfl(batch, party.local_embeddings, self.global_V)
            
            noise_scale = self._compute_noise_scale_round1()
            noise = np.random.normal(0, noise_scale, grad_V.shape)
            
            noisy_grad = grad_V + noise
            noisy_updates.append(noisy_grad)
            survivors.append(i)
        
        return noisy_updates, survivors
    
    def _secure_aggregation_round2(self, survivors_round1, noisy_updates_round1):
        replacement_noises = []
        survivors_round2 = []
        
        for idx, i in enumerate(survivors_round1):
            if np.random.random() < 0.1:
                continue
            
            party = self.parties[i]
            if len(party.local_data) == 0:
                continue
            
            batch = party._sample_batch(1)
            _, grad_V = party._compute_gradients_hfl(batch, party.local_embeddings, self.global_V)
            
            noise_scale_round1 = self._compute_noise_scale_round1()
            old_noise = np.random.normal(0, noise_scale_round1, grad_V.shape)
            
            noise_scale_round2 = self._compute_noise_scale_round2(len(survivors_round1))
            new_noise = np.random.normal(0, noise_scale_round2, grad_V.shape)
            
            replacement_noise = -old_noise + new_noise
            replacement_noises.append(replacement_noise)
            survivors_round2.append(i)
        
        return replacement_noises, survivors_round2
    
    def _compute_noise_scale_round1(self):
        sensitivity = self.dp.compute_sensitivity(self.privacy_type)
        noise_multiplier = self.dp.compute_noise_multiplier(
            self.epsilon, 1.0/self.n_users, self.T, self.delta
        )
        zeta1 = 1 / ((1 - self.omega) * self.n_users)
        return sensitivity * noise_multiplier * np.sqrt(zeta1)
    
    def _compute_noise_scale_round2(self, n_survivors_round1):
        sensitivity = self.dp.compute_sensitivity(self.privacy_type)
        noise_multiplier = self.dp.compute_noise_multiplier(
            self.epsilon, 1.0/self.n_users, self.T, self.delta
        )
        zeta2 = 1 / n_survivors_round1
        return sensitivity * noise_multiplier * np.sqrt(zeta2)
    
    def train(self, party_data_list, T, T_prime, learning_rate, sampling_rate=0.1):

        dropout_rate = 0.0
        
        if self.privacy_type == 'per_user':
            trimmed_data = []
            for i in range(self.n_users):
                trimmed = self.dp.apply_random_trimming(party_data_list[i], self.theta)
                trimmed_data.append(trimmed)
                self.parties[i].local_data = trimmed
            party_data_list = trimmed_data
        
        for t in range(T):
            n_active_round1 = int(self.n_users * (1 - dropout_rate))
            active_indices = np.random.choice(self.n_users, n_active_round1, replace=False)
            
            noisy_updates_round1, survivors_round1 = self._secure_aggregation_round1(active_indices)
            
            if len(survivors_round1) < (1 - self.omega) * self.n_users:
                logger.info(f"Round {t+1}: Too many dropouts, skipping")
                continue
            
            aggregated_grad_round1 = np.sum(noisy_updates_round1, axis=0) if noisy_updates_round1 else 0
            
            replacement_noises, survivors_round2 = self._secure_aggregation_round2(survivors_round1, noisy_updates_round1)
            
            aggregated_replacement = np.sum(replacement_noises, axis=0) if replacement_noises else 0
            
            if len(survivors_round2) > 0:
                total_update = aggregated_grad_round1 + aggregated_replacement
                self.global_V -= learning_rate * total_update / len(survivors_round1)
                self.global_V = self._clip_embeddings(self.global_V)
                self.coordinator.global_embeddings = self.global_V
            
            for party in self.parties:
                party.receive_global_embeddings(self.global_V)
            
            total_loss = 0
            n_with_data = 0
            for i in range(self.n_users):
                if len(party_data_list[i]) > 0:
                    loss = self.parties[i]._compute_loss_hfl(party_data_list[i], 
                                                        self.parties[i].local_embeddings,
                                                        self.global_V)
                    total_loss += loss
                    n_with_data += 1
            
            if n_with_data > 0:
                avg_loss = total_loss / n_with_data
                self.loss_history.append(avg_loss)
                logger.info(f"Round {t+1}/{self.T}, Active: {n_active_round1}, Loss: {avg_loss:.4f}")
            
            if(t / T <= 0.8):
                learning_rate *= 0.98
        
        for i in range(self.n_users):
            party = self.parties[i]
            if len(party_data_list[i]) > 0:
                party.fine_tune(20, learning_rate)
        
        return self.loss_history
    
    def predict(self, user_index, item_index):
        if user_index < self.n_users:
            party = self.parties[user_index]
            return party.predict(user_index, item_index)
        return 0.0
    
    def embedding_clip_unified(embeddings, R_max, item_embedding=False):
        embeddings = np.maximum(embeddings, 0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        if item_embedding:
            max_norm = np.sqrt(R_max)
        else:
            max_norm = np.sqrt(R_max)
        
        scaling = np.minimum(1, max_norm / (norms + 1e-8))
        return embeddings * scaling