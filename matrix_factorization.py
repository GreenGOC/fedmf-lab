
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MatrixFactorizer:
    def __init__(self, n_users, n_items, embedding_dim=20, R_max=20.0):
        self.R_max = R_max
        self.sqrt_R = np.sqrt(R_max)
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        scale = np.sqrt(R_max / embedding_dim)
        self.U = np.random.rand(n_users, embedding_dim) * scale
        self.V = np.random.rand(n_items, embedding_dim) * scale
        
        self.U = self._clip_embedding(self.U)
        self.V = self._clip_embedding(self.V)
        
        self.loss_history = []
        logger.info(f"Initialized MatrixFactorizer with {n_users} users, {n_items} items, embedding_dim={embedding_dim}, R_max={R_max}")
        
        
    def _clip_embedding(self, E):
        E = np.maximum(E, 0)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        scaling = np.minimum(1, self.sqrt_R / (norms + 1e-8))
        return E * scaling
    
    def _clip_single_embedding(self, e):
        e = np.maximum(e, 0)
        norm = np.linalg.norm(e)
        if(norm > self.sqrt_R):
            e = e * self.sqrt_R / norm
        return e
    
    def predict(self, user_index, item_index):
        if(user_index >= self.n_users or item_index >= self.n_items):
            raise IndexError(f"Indices out of bound in Predict")
        
        return np.dot(self.U[user_index], self.V[item_index])
    
    def predict_batch(self, indices) :
        predictions = []
        for user_index, item_index in indices:
            predictions.append(self.predict(user_index, item_index))
        return np.array(predictions)
    
    def loss(self, ratings_batch):
        total_loss = 0
        for user_index, item_index, rating in ratings_batch:
            pred = self.predict(user_index, item_index)
            total_loss += (rating - pred) ** 2
        return total_loss / len(ratings_batch)
    
    def compute_gradients(self, user_index, item_index, rating):
        prediction = self.predict(user_index, item_index)
        error = rating - prediction
        
        gradient_u = -2 * error * self.V[item_index]
        gradient_v = -2 * error * self.U[user_index]
        
        return gradient_u, gradient_v
    
    def compute_gradient_batch(self, batch):
        gradients_U = np.zeros_like(self.U)
        gradients_V = np.zeros_like(self.V)
        
        for user_index, item_index, rating in batch:
            gradient_u, gradient_v = self.compute_gradients(user_index, item_index, rating)
            gradients_U[user_index] += gradient_u
            gradients_V[item_index] += gradient_v
            
        return gradients_U, gradients_V
    
    def update_single(self, user_index, item_index, rating, learning_rate = 0.01):
        gradient_u, gradient_v = self.compute_gradients(user_index, item_index)
        
        self.U[user_index] -= learning_rate * gradient_u
        self.V[item_index] -= learning_rate * gradient_v
        
        self.U[user_index] = self._clip_single_embedding(self.U[user_index])
        self.V[item_index] = self._clip_single_embedding(self.V[item_index])
        
    def update_batch(self, batch, learning_rate = 0.01):
        gradients_U, gradients_V = self.compute_gradient_batch(batch)
        
        batch_size = len(batch)
        
        if(batch_size > 0):
            gradients_U /= batch_size
            gradients_V /= batch_size
            
        self.U -= learning_rate * gradients_U
        self.V -= learning_rate * gradients_V
        
        self.U = self._clip_embedding(self.U)
        self.V = self._clip_embedding(self.V)
        
        batch_loss = self.loss(batch)
        self.loss_history.append(batch_loss)
        
    def train(self, ratings, epochs = 100,
              batch_size = 16, learning_rate = 0.01, val_ratings = None, verbose = True):
        history = {
            'train_loss': [],
            'test_loss': [],
            'epochs': []
        }
        
        n_ratings = len(ratings)
        for epoch in range(epochs):
            shuffled_index = np.random.permutation(n_ratings)
            shuffled_ratings = [ratings[i] for i in shuffled_index]
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_ratings, batch_size):
                batch = shuffled_ratings[i:min(i + batch_size, n_ratings)]
                batch_loss = self.update_batch(batch, learning_rate)
                epoch_loss += batch_loss
                n_batches += 1
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            history['train_loss'].append(avg_loss)
            history['epochs'].append(epoch)
            
            if(val_ratings):
                val_loss = self.loss(val_ratings)
                history['val_loss'].append(val_loss)
                
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f}"
                if(val_ratings):
                    msg += f", Val Loss: {val_loss:.4f}"
                logger.info(msg)
        return history
    
    def evaluate(self, test_ratings):
        predictions = []
        true_ratings = []
        for user_index, item_index, rating in test_ratings:
            pred = self.predict(user_index)
            predictions.append(pred)
            true_ratings.append(rating)
            
        predictions = np.array(predictions)
        true_ratings = np.array(true_ratings)
        
        mse = np.mean((true_ratings - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_ratings - predictions))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'avg_prediction': np.mean(predictions),
            'std_predictions': np.std(predictions)
        }
        
    def get_embeddings(self):
        return self.U.copy(), self.V.copy()
    