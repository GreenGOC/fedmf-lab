

import numpy as np

class Party:
    def __init__(self, party_id, local_data, setting, embedding_dim, R_max):
        self.party_id = party_id
        self.local_data = local_data
        self.setting = setting
        self.embedding_dim = embedding_dim
        self.R_max = R_max
        
        if setting == 'VFL':
            self.local_items = list(set([item_index for _, item_index, _ in local_data]))
            self.local_embeddings = np.random.rand(len(self.local_items), embedding_dim) 
            self.local_users = None
            self.item_to_index = {item: index for index, item in enumerate(self.local_items)}
            self.user_to_index = None
            
        else:
            self.local_users = list(set([user_index for user_index, _, _ in local_data]))
            self.local_embeddings = np.random.rand(len(self.local_users), embedding_dim)
            self.local_items = None
            self.user_to_index = {user: index for index, user in enumerate(self.local_users)}
            self.item_to_index = None
            
        self.local_embeddings = self._clip_embeddings(self.local_embeddings)
        self.global_embeddings = None
        self.unique_users = list(set([u for u, _, _ in local_data])) if local_data else []
        self.unique_items = list(set([i for _, i, _ in local_data])) if local_data else []
        
    def _clip_embeddings(self, embeddings):
        embeddings = np.maximum(embeddings, 0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        scaling = np.minimum(1, np.sqrt(self.R_max) / (norms + 1e-8))
        return embeddings * scaling
    
    def receive_global_embeddings(self, global_embeddings):
        self.global_embeddings = global_embeddings.copy()
        
    def train_local(self, T_prime, learning_rate, dp=None, noise_multiplier=None, privacy_type='per_rating', sampling_rate=None, sampling_level='rating'):
        if self.setting == 'VFL':
            return self._train_vfl(T_prime, learning_rate, dp, noise_multiplier, privacy_type, sampling_rate, sampling_level)
        else:
            return self._train_hfl(T_prime, learning_rate, dp, noise_multiplier, privacy_type, sampling_rate, sampling_level)
        
    def _train_vfl(self, T_prime, learning_rate, dp, noise_multiplier, privacy_type, sampling_rate, sampling_level):
        if self.global_embeddings is None:
            raise ValueError("Global Embedding is None")
        
        U = self.global_embeddings.copy()
        V = self.local_embeddings.copy()
        
        for _ in range(T_prime):
            if sampling_rate:
                batch = self._sample_by_rate(sampling_rate, sampling_level)
            else:
                batch = self._sample_batch(32)
            
            gradient_U, gradient_V = self._compute_gradients_vfl(batch, U, V)
            
            if dp and noise_multiplier:
                sensitivity = dp.compute_sensitivity(privacy_type)
                noise_U = dp.gaussian_noise(gradient_U.shape, sensitivity, noise_multiplier)
                noise_V = dp.gaussian_noise(gradient_V.shape, sensitivity, noise_multiplier)
                gradient_U += noise_U
                gradient_V += noise_V
                
            U -= learning_rate * gradient_U
            V -= learning_rate * gradient_V
            
            U = self._clip_embeddings(U)
            V = self._clip_embeddings(V)
            
        self.local_embeddings = V
        loss = self._compute_loss_vfl(self.local_data, U, V)
        return U, loss
    
    def _train_hfl(self, T_prime, learning_rate, dp, noise_multiplier, privacy_type, sampling_rate, sampling_level):
        if self.global_embeddings is None:
            raise ValueError("Global Embedding is None")
        
        U = self.local_embeddings.copy()
        V = self.global_embeddings.copy()
        
        for _ in range(T_prime):
            if sampling_rate:
                batch = self._sample_by_rate(sampling_rate, sampling_level)
            else:
                batch = self._sample_batch(32)
            
            gradient_U, gradient_V = self._compute_gradients_hfl(batch, U, V)
            
            if dp and noise_multiplier:
                sensitivity = dp.compute_sensitivity(privacy_type)
                noise_V = dp.gaussian_noise(gradient_V.shape, sensitivity, noise_multiplier)
                gradient_V += noise_V
                
            U -= learning_rate * gradient_U
            V -= learning_rate * gradient_V
            
            U = self._clip_embeddings(U)
            V = self._clip_embeddings(V)
            
        self.local_embeddings = U
        loss = self._compute_loss_hfl(self.local_data, U, V)
        return V, loss
    
    def _pretrain_step(self, learning_rate, sampling_rate=None):
        if self.setting == 'HFL' or self.setting == 'LFL':
            if self.global_embeddings is None:
                return 
            U = self.local_embeddings.copy()
            V = self.global_embeddings.copy()
            
            if sampling_rate:
                batch = self._sample_by_rate(sampling_rate, 'rating')
            else:
                batch = self._sample_batch(32)
            
            gradient_U, _ = self._compute_gradients_hfl(batch, U, V)
            
            U -= learning_rate * gradient_U
            U = self._clip_embeddings(U)
            
            self.local_embeddings = U
            
    def fine_tune(self, fine_tune_iters, learning_rate):
        if self.setting == 'VFL':
            self._fine_tune_vfl(fine_tune_iters, learning_rate)
        else:
            self._fine_tune_hfl(fine_tune_iters, learning_rate)
    
    def _fine_tune_vfl(self, kappa, learning_rate):
        if self.global_embeddings is None or not self.local_data:
            return
        
        V = self.local_embeddings.copy()
        
        for _ in range(kappa):
            grad_V = np.zeros_like(V)
            
            for user_index, item_index, rating in self.local_data:
                if item_index in self.item_to_index:
                    local_item_index = self.item_to_index[item_index]
                    u = self.global_embeddings[user_index]
                    v = V[local_item_index]
                    
                    pred = np.dot(u, v)
                    error = rating - pred
                    grad_V[local_item_index] += -2 * error * u
            
            if len(self.local_data) > 0:
                grad_V = grad_V / len(self.local_data)
            
            V -= learning_rate * grad_V
            V = self._clip_embeddings(V)
        
        self.local_embeddings = V
    
    def _fine_tune_hfl(self, fine_tune_iters, learning_rate):
        if self.global_embeddings is None or not self.local_data:
            return
        
        U = self.local_embeddings.copy()
        V = self.global_embeddings.copy()
        
        for _ in range(fine_tune_iters):
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            
            for user_index, item_index, rating in self.local_data:
                if user_index in self.user_to_index:
                    local_user_index = self.user_to_index[user_index]
                    u = U[local_user_index]
                    v = V[item_index]
                    
                    pred = np.dot(u, v)
                    error = rating - pred
                    
                    grad_U[local_user_index] += -2 * error * v
                    grad_V[item_index] += -2 * error * u
            
            if len(self.local_data) > 0:
                grad_U = grad_U / len(self.local_data)
                grad_V = grad_V / len(self.local_data)
            
            U -= learning_rate * grad_U
            V -= learning_rate * grad_V
            
            U = self._clip_embeddings(U)
            V = self._clip_embeddings(V)
        
        self.local_embeddings = U
        self.global_embeddings = V
    
    def _compute_gradients_vfl(self, batch, U, V):
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        
        for user_index, item_index, rating in batch:
            if item_index in self.item_to_index:
                local_item_index = self.item_to_index[item_index]
                u = U[user_index]
                v = V[local_item_index]
                
                pred = np.dot(u, v)
                error = rating - pred
                
                grad_U[user_index] += -2 * error * v
                grad_V[local_item_index] += -2 * error * u
        
        if len(batch) > 0:
            grad_U = grad_U / len(batch)
            grad_V = grad_V / len(batch)
        
        return grad_U, grad_V
    
    def _compute_gradients_hfl(self, batch, U, V):
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        
        for user_index, item_index, rating in batch:
            if user_index in self.user_to_index:
                local_user_index = self.user_to_index[user_index]
                u = U[local_user_index]
                v = V[item_index]
                
                pred = np.dot(u, v)
                error = rating - pred
                
                grad_U[local_user_index] += -2 * error * v
                grad_V[item_index] += -2 * error * u
        
        if len(batch) > 0:
            grad_U = grad_U / len(batch)
            grad_V = grad_V / len(batch)
        
        return grad_U, grad_V
    
    def _compute_loss_vfl(self, data, U, V):
        total_loss = 0
        for user_index, item_index, rating in data:
            if item_index in self.item_to_index:
                local_item_index = self.item_to_index[item_index]
                pred = np.dot(U[user_index], V[local_item_index])
                total_loss += (rating - pred) ** 2
        return total_loss / len(data) if data else 0
    
    def _compute_loss_hfl(self, data, U, V):
        total_loss = 0
        for user_index, item_index, rating in data:
            if user_index in self.user_to_index:
                local_user_index = self.user_to_index[user_index]
                pred = np.dot(U[local_user_index], V[item_index])
                total_loss += (rating - pred) ** 2
        return total_loss / len(data) if data else 0
    
    def _sample_batch(self, batch_size):
        if len(self.local_data) <= batch_size:
            return self.local_data
        indices = np.random.choice(len(self.local_data), batch_size, replace=False)
        return [self.local_data[i] for i in indices]
    
    def _sample_by_rate(self, sampling_rate, level='rating'):
        if level == 'user' and self.unique_users:
            n_users = max(1, int(sampling_rate * len(self.unique_users)))
            selected_users = np.random.choice(self.unique_users, n_users, replace=False)
            selected_users_set = set(selected_users)
            batch = [d for d in self.local_data if d[0] in selected_users_set]
        else:
            n_ratings = max(1, int(sampling_rate * len(self.local_data)))
            indices = np.random.choice(len(self.local_data), n_ratings, replace=False)
            batch = [self.local_data[i] for i in indices]
        return batch
    
    def predict(self, user_index, item_index):
        if self.setting == 'VFL':
            if item_index in self.item_to_index:
                local_item_index = self.item_to_index[item_index]
                if self.global_embeddings is not None:
                    return np.dot(self.global_embeddings[user_index], self.local_embeddings[local_item_index])
        else:
            if user_index in self.user_to_index:
                local_user_index = self.user_to_index[user_index]
                if self.global_embeddings is not None:
                    return np.dot(self.local_embeddings[local_user_index], self.global_embeddings[item_index])
        return 0