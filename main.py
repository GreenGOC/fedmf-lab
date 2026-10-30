import numpy as np
import pandas as pd
import logging
import time
import pickle
import os
import sys
import matplotlib.pyplot as plt

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    BASE_DIR,
    os.path.join(BASE_DIR, 'model'),
    os.path.join(BASE_DIR, 'nodes'),
    os.path.join(BASE_DIR, 'privacy'),
    os.path.join(BASE_DIR, 'utils')
])

from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_partitioner import DataPartitioner
from sgdmf_vfl import VFL_SGDMF
from sgdmf_hfl import HFL_SGDMF
from sgdmf_lfl import LFL_SGDMF
from fed_avg import FedAvg
from fed_prox import FedProx
from fed_admm import FedADMM
from differential_privacy import DifferentialPrivacy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    name: str
    path: str
    n_users: int
    n_items: int
    rating_range: Tuple[float, float]
    
@dataclass
class ExperimentConfig:
    experiment_id: str
    algorithm: str
    setting: str
    dataset: str
    n_parties: int
    embedding_dim: int
    partition_method: str = 'random'
    R_max: float = 5.0
    T: int = 100
    T_prime: int = 10
    learning_rate: float = 0.05
    sample_fraction: float = 0.1
    epsilon: float = 1.0
    delta: float = 1e-5
    privacy_type: str = 'per_rating'
    theta: int = 5
    mu: float = 0.0001
    rho: float = 1.0
    omega: float = 0.3
    test_ratio: float = 0.2
    random_seed: int = 42

@dataclass
class AttackerConfig:
    run_attacks: bool = True
    shadow_models_count: int = 5
    attack_epochs: int = 15
    attack_learning_rate: float = 0.1
    attack_train_ratio: float = 0.8
    sample_size_for_attack: int = 200

class DatasetManager:
    def __init__(self, data_dir='data'):
        self.datasets: Dict[str, Tuple[List, int, int]] = {}
        self.train_test_splits: Dict[str, Tuple[List, List]] = {}
        self.data_dir = data_dir
        self.dataset_configs = {
            'MovieLens1M': DatasetConfig(
                name='MovieLens 1M',
                path=os.path.join(data_dir, 'MovieLens1M', 'ratings.csv'),
                n_users=6040,
                n_items=3952,
                rating_range=(0.5, 5.0)
            ),
            'synthetic': DatasetConfig(
                name='Synthetic Dataset',
                path='',
                n_users=3000,
                n_items=500,
                rating_range=(0.5, 5.0)
            ),
            'MovieLens10M': DatasetConfig(
                name='MovieLens 10M',
                path=os.path.join(data_dir, 'MovieLens10M', 'ratings.csv'),
                n_users=71567,
                n_items=10681,
                rating_range=(0.5, 5.0)
            ),
            'MovieLens25M': DatasetConfig(
                name='MovieLens 25M',
                path=os.path.join(data_dir, 'MovieLens25M', 'ratings.csv'),
                n_users=162541,
                n_items=59047,
                rating_range=(0.5, 5.0)
            )
        }
        
    def _get_cache_key(self, dataset_name: str, test_ratio: float, random_seed: int) -> str:
        key_str = f"{dataset_name}_{test_ratio}_{random_seed}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_partition_key(self, dataset_name: str, test_ratio: float, random_seed: int, 
                          setting: str, n_parties: int, partition_method: str) -> str:
        key_str = f"{dataset_name}_{test_ratio}_{random_seed}_{setting}_{n_parties}_{partition_method}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def load_dataset(self, dataset_name: str, cache_dir: str = 'cache') -> Tuple[List, int, int]:
        if dataset_name in self.datasets:
            return self.datasets[dataset_name]
            
        config = self.dataset_configs[dataset_name]
        cache_file = os.path.join(cache_dir, f"{dataset_name}_processed.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            self.datasets[dataset_name] = data
            return data
        
        if dataset_name == 'synthetic':
            data = self._generate_iid_synthetic_data(config.n_users, config.n_items)
            self.datasets[dataset_name] = data
            return data
            
        try:
            if os.path.exists(config.path):
                df = pd.read_csv(config.path)
                
                column_map = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'user' in col_lower:
                        column_map['user'] = col
                    elif 'item' in col_lower or 'movie' in col_lower:
                        column_map['item'] = col
                    elif 'rating' in col_lower or 'score' in col_lower:
                        column_map['rating'] = col
                
                if len(column_map) >= 3:
                    df_clean = pd.DataFrame({
                        'user_id': df[column_map['user']],
                        'item_id': df[column_map['item']],
                        'rating': df[column_map['rating']]
                    })
                else:
                    df_clean = pd.DataFrame({
                        'user_id': df.iloc[:, 0],
                        'item_id': df.iloc[:, 1],
                        'rating': df.iloc[:, 2]
                    })
                
                user_min = df_clean['user_id'].min()
                item_min = df_clean['item_id'].min()
                
                ratings = list(zip(
                    df_clean['user_id'].values - user_min,
                    df_clean['item_id'].values - item_min,
                    df_clean['rating'].values
                ))
                
                n_users = df_clean['user_id'].max() - user_min + 1
                n_items = df_clean['item_id'].max() - item_min + 1
                
            else:
                data = self._generate_iid_synthetic_data(config.n_users, config.n_items)
                self.datasets[dataset_name] = data
                return data
                
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            data = self._generate_iid_synthetic_data(config.n_users, config.n_items)
            self.datasets[dataset_name] = data
            return data
        
        os.makedirs(cache_dir, exist_ok=True)
        
        data = (ratings, n_users, n_items)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
            
        self.datasets[dataset_name] = data
        return data
    
    def get_train_test_split(self, dataset_name: str, test_ratio: float = 0.2, 
                            random_seed: int = 42, cache_dir: str = 'cache') -> Tuple[List, List]:
        cache_key = self._get_cache_key(dataset_name, test_ratio, random_seed)
        
        if cache_key in self.train_test_splits:
            return self.train_test_splits[cache_key]
        
        cache_file = os.path.join(cache_dir, f"split_{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                train_ratings, test_ratings = pickle.load(f)
            self.train_test_splits[cache_key] = (train_ratings, test_ratings)
            return train_ratings, test_ratings
        
        ratings, n_users, n_items = self.load_dataset(dataset_name, cache_dir)
        
        np.random.seed(random_seed)
        np.random.shuffle(ratings)
        
        split_idx = int(len(ratings) * (1 - test_ratio))
        train_ratings = ratings[:split_idx]
        test_ratings = ratings[split_idx:]
        
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump((train_ratings, test_ratings), f)
        
        self.train_test_splits[cache_key] = (train_ratings, test_ratings)
        return train_ratings, test_ratings
    
    def get_partitioned_data(self, dataset_name: str, setting: str, n_parties: int,
                            test_ratio: float = 0.2, random_seed: int = 42,
                            partition_method: str = 'random', cache_dir: str = 'cache') -> Tuple[List, List, int, int]:
        cache_key = self._get_partition_key(dataset_name, test_ratio, random_seed, 
                                           setting, n_parties, partition_method)
        
        partition_cache_file = os.path.join(cache_dir, f"partition_{cache_key}.pkl")
        
        if os.path.exists(partition_cache_file):
            with open(partition_cache_file, 'rb') as f:
                party_data_list, party_maps, n_users, n_items = pickle.load(f)
            return party_data_list, party_maps, n_users, n_items
        
        train_ratings, test_ratings = self.get_train_test_split(
            dataset_name, test_ratio, random_seed, cache_dir
        )
        
        _, n_users, n_items = self.load_dataset(dataset_name, cache_dir)
        
        try:
            if setting == 'VFL':
                party_data_list, party_maps = DataPartitioner.partition_vertical(
                    train_ratings, n_parties, n_items, split_by=partition_method
                )
            elif setting == 'HFL':
                party_data_list, party_maps = DataPartitioner.partition_horizontal(
                    train_ratings, n_parties, n_users, split_by=partition_method
                )
            elif setting == 'LFL':
                party_data_list, party_maps = DataPartitioner.partition_local(train_ratings, n_users)
                n_parties = n_users
            else:
                raise ValueError(f"Unknown setting: {setting}")
        except Exception as e:
            logger.warning(f"Partitioning failed, using simple partition: {e}")
            party_data_list = self._simple_partition(train_ratings, n_parties, setting)
            party_maps = []
        
        os.makedirs(cache_dir, exist_ok=True)
        with open(partition_cache_file, 'wb') as f:
            pickle.dump((party_data_list, party_maps, n_users, n_items), f)
        
        return party_data_list, party_maps, n_users, n_items
    
    def _generate_iid_synthetic_data(self, n_users: int, n_items: int, 
                                   density: float = 0.2) -> Tuple[List, int, int]:
        np.random.seed(42)
        
        user_embeddings = np.random.randn(n_users, 20)  
        item_embeddings = np.random.randn(n_items, 20) 
        
        n_ratings = int(n_users * n_items * density)

        total_pairs = n_users * n_items
        
        if n_ratings > total_pairs:
            n_ratings = total_pairs
            
        indices = np.random.choice(total_pairs, n_ratings, replace=False)
        user_indices = indices // n_items
        item_indices = indices % n_items
        
        ratings = []
        
        for i, j in zip(user_indices, item_indices):
            true_rating = np.dot(user_embeddings[i], item_embeddings[j])
            
            rating_scale = 2.0  
            noise_std = 0.5    
            
            rating = true_rating * rating_scale + np.random.normal(0, noise_std)
            rating = np.clip(rating, 0.5, 5.0)
            
            ratings.append((int(i), int(j), float(rating)))
        
        np.random.shuffle(ratings)
        
        logger.info(f"Generated {len(ratings)} IID ratings ({100*len(ratings)/(n_users*n_items):.2f}% density)")
        logger.info(f"Rating stats: min={min(r for _,_,r in ratings):.2f}, "
                    f"max={max(r for _,_,r in ratings):.2f}, "
                    f"mean={np.mean([r for _,_,r in ratings]):.2f}")
        
        return (ratings, n_users, n_items)
    
    def _simple_partition(self, ratings: List[Tuple], n_parties: int, setting: str) -> List[List[Tuple]]:
        np.random.shuffle(ratings)
        party_data_list = []
        
        if setting == 'VFL':
            items = list(set([item for _, item, _ in ratings]))
            np.random.shuffle(items)
            item_groups = np.array_split(items, n_parties)
            
            for item_group in item_groups:
                party_data = [r for r in ratings if r[1] in set(item_group)]
                party_data_list.append(party_data)
        else:
            chunks = np.array_split(ratings, n_parties)
            party_data_list = [list(chunk) for chunk in chunks]
        
        return party_data_list

class PrivacyAttacker:
    @dataclass
    class ShadowModelConfig:
        dataset_name: str
        n_users: int
        n_items: int
        embedding_dim: int
        R_max: float = 5.0
        train_ratio: float = 0.8
        shadow_models_count: int = 10
        attack_epochs: int = 20
        learning_rate: float = 0.02
    
    def __init__(self, shadow_config: ShadowModelConfig):
        self.config = shadow_config
        self.shadow_models = []
        self.shadow_datasets = []
        self.attack_model = None
        self.metadata = {}
        
    def train_shadow_models(self, ratings: List[Tuple]):
        np.random.shuffle(ratings)
        train_size = int(len(ratings) * self.config.train_ratio)
        
        for i in range(self.config.shadow_models_count):
            shadow_train = ratings[:train_size]
            shadow_test = ratings[train_size:]
            
            U = np.random.rand(self.config.n_users, self.config.embedding_dim)
            V = np.random.rand(self.config.n_items, self.config.embedding_dim)
            
            U, V = self._train_matrix_factorization(shadow_train, U, V)
            
            shadow_data = self._extract_shadow_data(shadow_train, shadow_test, U, V)
            self.shadow_datasets.append(shadow_data)
            
            self.shadow_models.append((U, V))
            
    def _train_matrix_factorization(self, ratings: List[Tuple], U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        learning_rate = self.config.learning_rate
        
        for epoch in range(self.config.attack_epochs):
            total_loss = 0
            
            for user_idx, item_idx, rating in ratings:
                prediction = np.dot(U[user_idx], V[item_idx])
                error = rating - prediction
                
                U[user_idx] += learning_rate * error * V[item_idx]
                V[item_idx] += learning_rate * error * U[user_idx]
                
                total_loss += error ** 2
                
        U = self._clip_embeddings(U)
        V = self._clip_embeddings(V)
        
        return U, V
    
    def _clip_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        scaling = np.minimum(1, np.sqrt(self.config.R_max) / (norms + 1e-8))
        return embeddings * scaling
    
    def _extract_shadow_data(self, train_ratings: List[Tuple], test_ratings: List[Tuple], 
                           U: np.ndarray, V: np.ndarray) -> Dict[str, Any]:
        train_features = []
        train_labels = []
        
        for user_idx, item_idx, rating in train_ratings:
            pred = np.dot(U[user_idx], V[item_idx])
            error = abs(rating - pred)
            user_norm = np.linalg.norm(U[user_idx])
            item_norm = np.linalg.norm(V[item_idx])
            dot_product = pred
            
            features = np.array([pred, error, user_norm, item_norm, dot_product, rating])
            train_features.append(features)
            train_labels.append(1)
        
        test_features = []
        test_labels = []
        
        test_sample = test_ratings[:len(train_ratings)]
        
        for user_idx, item_idx, rating in test_sample:
            pred = np.dot(U[user_idx], V[item_idx])
            error = abs(rating - pred)
            user_norm = np.linalg.norm(U[user_idx])
            item_norm = np.linalg.norm(V[item_idx])
            dot_product = pred
            
            features = np.array([pred, error, user_norm, item_norm, dot_product, rating])
            test_features.append(features)
            test_labels.append(0)
        
        all_features = np.vstack(train_features + test_features)
        all_labels = np.array(train_labels + test_labels)
        
        return {
            'features': all_features,
            'labels': all_labels,
            'train_size': len(train_ratings),
            'test_size': len(test_sample)
        }
    
    def build_attack_model(self):
        all_features = []
        all_labels = []
        
        for shadow_data in self.shadow_datasets:
            all_features.append(shadow_data['features'])
            all_labels.append(shadow_data['labels'])
        
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.attack_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.attack_model.fit(X_train, y_train)
        
        accuracy = self.attack_model.score(X_test, y_test)
        
        self.metadata['attack_model_accuracy'] = accuracy
        self.metadata['attack_model_features'] = X.shape[1]
        self.metadata['attack_model_samples'] = X.shape[0]
        
        return accuracy
    
    def perform_membership_inference(self, target_model: Any, target_ratings: List[Tuple]) -> Dict[str, Any]:
        if self.attack_model is None:
            raise ValueError("Attack model not trained.")
        
        target_predictions = []
        for user_idx, item_idx, rating in target_ratings:
            pred = target_model.predict(user_idx, item_idx)
            target_predictions.append((user_idx, item_idx, rating, pred))
        
        attack_features = []
        for user_idx, item_idx, rating, pred in target_predictions:
            error = abs(rating - pred)
            
            if hasattr(target_model, 'global_U'):
                user_norm = np.linalg.norm(target_model.global_U[user_idx])
                item_norm = 0
                if hasattr(target_model, 'party_V'):
                    for k in range(target_model.n_parties):
                        if item_idx in target_model.party_items[k]:
                            local_idx = target_model.party_items[k].index(item_idx)
                            item_norm = np.linalg.norm(target_model.party_V[k][local_idx])
                            break
                dot_product = pred
            else:
                user_norm = 0
                item_norm = 0
                dot_product = pred
            
            features = np.array([pred, error, user_norm, item_norm, dot_product, rating])
            attack_features.append(features)
        
        X_attack = np.vstack(attack_features)
        predictions = self.attack_model.predict(X_attack)
        probabilities = self.attack_model.predict_proba(X_attack)
        
        membership_scores = probabilities[:, 1]
        
        results = []
        for idx, (user_idx, item_idx, rating, pred) in enumerate(target_predictions):
            results.append({
                'user_idx': user_idx,
                'item_idx': item_idx,
                'rating': rating,
                'prediction': pred,
                'membership_score': membership_scores[idx],
                'is_member': predictions[idx] == 1
            })
        
        avg_membership_score = np.mean(membership_scores)
        member_count = np.sum(predictions == 1)
        
        attack_results = {
            'results': results,
            'summary': {
                'total_samples': len(target_ratings),
                'predicted_members': int(member_count),
                'predicted_non_members': int(len(target_ratings) - member_count),
                'avg_membership_score': float(avg_membership_score),
                'membership_rate': float(member_count / len(target_ratings))
            },
            'metadata': self.metadata
        }
        
        return attack_results
    
    def perform_privacy_risk_analysis(self, target_model: Any, 
                                    sample_ratings: List[Tuple]) -> Dict[str, Any]:
        inference_results = self.perform_membership_inference(target_model, sample_ratings)
        
        high_risk_threshold = 0.7
        medium_risk_threshold = 0.3
        
        high_risk = [r for r in inference_results['results'] if r['membership_score'] > high_risk_threshold]
        medium_risk = [r for r in inference_results['results'] 
                      if medium_risk_threshold <= r['membership_score'] <= high_risk_threshold]
        low_risk = [r for r in inference_results['results'] if r['membership_score'] < medium_risk_threshold]
        
        risk_distribution = {
            'high_risk': len(high_risk),
            'medium_risk': len(medium_risk),
            'low_risk': len(low_risk),
            'total': len(sample_ratings)
        }
        
        avg_prediction_error = np.mean([abs(r['rating'] - r['prediction']) for r in inference_results['results']])
        
        risk_score = (len(high_risk) * 1.0 + len(medium_risk) * 0.5) / len(sample_ratings)
        
        return {
            'risk_distribution': risk_distribution,
            'risk_score': float(risk_score),
            'avg_prediction_error': float(avg_prediction_error),
            'summary': inference_results['summary']
        }

class ExperimentRunner:
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.results = []
        
    def run_experiment(self, config: ExperimentConfig, attacker_config: Optional[AttackerConfig] = None) -> Dict[str, Any]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Experiment: {config.experiment_id}")
        
        start_time = time.time()
        
        party_data_list, party_maps, n_users, n_items = self.dataset_manager.get_partitioned_data(
            dataset_name=config.dataset,
            setting=config.setting,
            n_parties=config.n_parties,
            test_ratio=config.test_ratio,
            random_seed=config.random_seed,
            partition_method=config.partition_method
        )
        
        party_sizes = [len(data) for data in party_data_list]
        
        test_ratings = self.dataset_manager.get_train_test_split(
            dataset_name=config.dataset,
            test_ratio=config.test_ratio,
            random_seed=config.random_seed
        )[1]
        
        algorithm = self._initialize_algorithm(config, n_users, n_items, party_data_list)
        
        train_results = self._train_algorithm(algorithm, config, party_data_list)
        
        eval_results = self._evaluate_algorithm(algorithm, config, test_ratings, party_data_list)
        
        logger.info(f"\nValidation Results for {config.algorithm}:")
        logger.info(f"  MSE: {eval_results.get('mse', 'N/A'):.4f}")
        logger.info(f"  RMSE: {eval_results.get('rmse', 'N/A'):.4f}")
        logger.info(f"  MAE: {eval_results.get('mae', 'N/A'):.4f}")
        logger.info(f"  R² Score: {eval_results.get('r2_score', 'N/A'):.4f}")
        print(f"\nValidation Results for {config.algorithm}:")
        print(f"  MSE: {eval_results.get('mse', 'N/A'):.4f}")
        print(f"  RMSE: {eval_results.get('rmse', 'N/A'):.4f}")
        print(f"  MAE: {eval_results.get('mae', 'N/A'):.4f}")
        print(f"  R² Score: {eval_results.get('r2_score', 'N/A'):.4f}")
        
        attack_results = None
        if attacker_config and attacker_config.run_attacks:
            attack_results = self._run_privacy_attacks(
                algorithm, config, n_users, n_items, 
                party_data_list, test_ratings, attacker_config
            )
        
        experiment_time = time.time() - start_time
        
        result = {
            'experiment_id': config.experiment_id,
            'config': config,
            'train_results': train_results,
            'eval_results': eval_results,
            'attack_results': attack_results,
            'training_time': experiment_time,
            'party_data_sizes': party_sizes,
            'n_users': n_users,
            'n_items': n_items
        }
        
        if 'loss_history' in train_results and train_results['loss_history']:
            try:
                from evaluator import ModelEvaluator
                evaluator = ModelEvaluator()
                plot_dir = 'plots'
                os.makedirs(plot_dir, exist_ok=True)
                safe_id = config.experiment_id.replace('/', '_').replace('\\', '_').replace(':', '_')
                plot_path = os.path.join(plot_dir, f'training_loss_{safe_id}.png')
                
                plot_result = {
                    'experiment_id': config.experiment_id,
                    'config': config,
                    'train_results': train_results,
                    'eval_results': eval_results
                }
                evaluator.plot_training_curves(plot_result, save_path=plot_path)
                logger.info(f"Training loss plot saved to {plot_path}")
                result['plot_path'] = plot_path
            except Exception as e:
                logger.warning(f"Failed to save training loss plot: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        self.results.append(result)
        
        logger.info(f"Experiment completed in {experiment_time:.2f} seconds")
        if 'mse' in eval_results:
            logger.info(f"Test MSE: {eval_results['mse']:.4f}")
        if attack_results:
            attack_file = self._save_inference_attack_results(
                algorithm, config, attack_results, test_ratings
            )
            result['attack_file'] = attack_file
        
        val_file = self._save_validation_results(
            config, eval_results, train_results
        )
        result['validation_file'] = val_file
        
        return result
    
    def _run_privacy_attacks(self, algorithm, config: ExperimentConfig, n_users: int, n_items: int,
                           party_data_list: List, test_ratings: List, attacker_config: AttackerConfig) -> Dict[str, Any]:
        logger.info("Running privacy attacks on the trained model...")
        
        all_training_data = []
        for party_data in party_data_list:
            all_training_data.extend(party_data)
        
        if len(all_training_data) < 100 or len(test_ratings) < 100:
            logger.warning("Insufficient data for privacy attacks")
            return None
        
        np.random.shuffle(all_training_data)
        np.random.shuffle(test_ratings)
        
        sample_size = min(attacker_config.sample_size_for_attack, len(all_training_data), len(test_ratings))
        test_sample = test_ratings[:sample_size]
        
        combined_sample = test_sample
        
        shadow_config = PrivacyAttacker.ShadowModelConfig(
            dataset_name=config.dataset,
            n_users=n_users,
            n_items=n_items,
            embedding_dim=config.embedding_dim,
            R_max=config.R_max,
            shadow_models_count=attacker_config.shadow_models_count,
            attack_epochs=attacker_config.attack_epochs,
            learning_rate=attacker_config.attack_learning_rate,
            train_ratio=attacker_config.attack_train_ratio
        )
        
        attacker = PrivacyAttacker(shadow_config)
        
        try:
            attacker.train_shadow_models(all_training_data[:min(1000, len(all_training_data))])
            attack_accuracy = attacker.build_attack_model()
            
            risk_analysis = attacker.perform_privacy_risk_analysis(algorithm, combined_sample)
            
            attack_results = {
                'attack_model_accuracy': attack_accuracy,
                'risk_analysis': risk_analysis,
                'attacker_config': attacker_config
            }
            
            logger.info(f"Privacy Attack Results:")
            logger.info(f"  Attack Model Accuracy: {attack_accuracy:.4f}")
            logger.info(f"  Privacy Risk Score: {risk_analysis.get('risk_score', 0):.4f}")
            
            return attack_results
            
        except Exception as e:
            logger.error(f"Failed to run privacy attacks: {e}")
            return None
    
    def _initialize_algorithm(self, config: ExperimentConfig, n_users: int, n_items: int, 
                            party_data_list: List) -> Any:
        try:
            if config.algorithm == 'VFL-SGDMF':
                algorithm = VFL_SGDMF(
                    n_parties=config.n_parties,
                    n_users=n_users,
                    n_items=n_items,
                    embedding_dim=config.embedding_dim,
                    R_max=config.R_max,
                    epsilon=config.epsilon,
                    delta=config.delta,
                    privacy_type=config.privacy_type,
                    theta=config.theta
                )
                algorithm.initialize(party_data_list)
                
            elif config.algorithm == 'HFL-SGDMF':
                algorithm = HFL_SGDMF(
                    n_parties=config.n_parties,
                    n_users=n_users,
                    n_items=n_items,
                    embedding_dim=config.embedding_dim,
                    R_max=config.R_max,
                    epsilon=config.epsilon,
                    delta=config.delta,
                    privacy_type=config.privacy_type,
                    theta=config.theta,
                    T=config.T,
                    T_prime=config.T_prime
                )
                algorithm.initialize(party_data_list)
                
            elif config.algorithm == 'LFL-SGDMF':
                algorithm = LFL_SGDMF(
                    n_users=n_users,
                    n_items=n_items,
                    embedding_dim=config.embedding_dim,
                    R_max=config.R_max,
                    epsilon=config.epsilon,
                    delta=config.delta,
                    privacy_type=config.privacy_type,
                    theta=config.theta,
                    omega=config.omega,
                    T=config.T
                )
                algorithm.initialize(party_data_list)
                
            elif config.algorithm == 'FedAvg':
                algorithm = FedAvg(
                    n_parties=config.n_parties,
                    n_users=n_users,
                    n_items=n_items,
                    embedding_dim=config.embedding_dim,
                    R_max=config.R_max,
                    setting=config.setting
                )
                algorithm.initialize_parties(party_data_list)
                
            elif config.algorithm == 'FedProx':
                algorithm = FedProx(
                    n_parties=config.n_parties,
                    n_users=n_users,
                    n_items=n_items,
                    embedding_dim=config.embedding_dim,
                    R_max=config.R_max,
                    setting=config.setting,
                    mu=config.mu
                )
                algorithm.initialize_parties(party_data_list)
                
            elif config.algorithm == 'FedADMM':
                algorithm = FedADMM(
                    n_parties=config.n_parties,
                    n_users=n_users,
                    n_items=n_items,
                    embedding_dim=config.embedding_dim,
                    R_max=config.R_max,
                    setting=config.setting,
                    rho=config.rho
                )
                algorithm.initialize_parties(party_data_list)
                
            else:
                raise ValueError(f"Unknown algorithm: {config.algorithm}")
            
            return algorithm
            
        except Exception as e:
            logger.error(f"Error initializing algorithm {config.algorithm}: {e}")
            raise
    
    def _train_algorithm(self, algorithm, config: ExperimentConfig, party_data_list: List) -> Dict:
        train_results = {}
        
        try:
            if config.algorithm in ['VFL-SGDMF', 'HFL-SGDMF', 'LFL-SGDMF']:
                loss_history = algorithm.train(
                    party_data_list=party_data_list,
                    T=config.T,
                    T_prime=config.T_prime,
                    learning_rate=config.learning_rate,
                    sampling_rate=0.1
                )
                train_results['loss_history'] = loss_history
                train_results['final_loss'] = loss_history[-1] if loss_history else None
                
            elif config.algorithm in ['FedAvg', 'FedProx', 'FedADMM']:
                algorithm.train(
                    party_data_list=party_data_list,
                    T=config.T,
                    local_epochs=config.T_prime,
                    learning_rate=config.learning_rate,
                    sample_fraction=config.sample_fraction
                )
                # Capture loss history from algorithms
                if hasattr(algorithm, 'loss_history') and algorithm.loss_history:
                    train_results['loss_history'] = algorithm.loss_history
                    train_results['final_loss'] = algorithm.loss_history[-1] if algorithm.loss_history else None
                else:
                    train_results['final_loss'] = None
                    train_results['loss_history'] = []
                
        except Exception as e:
            train_results['error'] = str(e)
            train_results['final_loss'] = None
            logger.error(f"Error during training: {e}")
        
        return train_results
    
    def _evaluate_algorithm(self, algorithm, config: ExperimentConfig, test_ratings: List,
                          party_data_list: List) -> Dict:
        from evaluator import ModelEvaluator
        evaluator = ModelEvaluator()
        
        predictions = []
        true_ratings = []
        
        n_eval = min(100, len(test_ratings))
        if len(test_ratings) > 0:
            eval_subset = np.random.choice(len(test_ratings), n_eval, replace=False)
        else:
            return {'mse': float('inf'), 'rmse': float('inf'), 'mae': float('inf')}
        
        for idx in eval_subset:
            user_idx, item_idx, rating = test_ratings[idx]
            try:
                pred = algorithm.predict(user_idx, item_idx)
                predictions.append(pred)
                true_ratings.append(rating)
            except Exception as e:
                predictions.append(0.0)
                true_ratings.append(rating)
        
        predictions = np.array(predictions)
        true_ratings = np.array(true_ratings)
        
        metrics = evaluator.calculate_metrics(predictions, true_ratings)
        
        # Store predictions and true values for plotting
        metrics['predictions'] = predictions
        metrics['true_values'] = true_ratings
        
        try:
            if hasattr(algorithm, 'coordinator'):
                comm_stats = algorithm.coordinator.get_communication_stats()
                metrics.update(comm_stats)
        except:
            pass
        
        return metrics
    
    def _save_inference_attack_results(self, algorithm, config, attack_results, test_ratings, output_dir='inference_attacks'):
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_id = config.experiment_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        attack_file = os.path.join(output_dir, f'inference_attack_{safe_id}_{timestamp}.txt')
        
        with open(attack_file, 'w') as f:
            f.write(f"Inference Attack Results for Experiment: {config.experiment_id}\n")
            f.write("="*80 + "\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  Algorithm: {config.algorithm}\n")
            f.write(f"  Setting: {config.setting}\n")
            f.write(f"  Dataset: {config.dataset}\n")
            f.write(f"  Parties: {config.n_parties}\n")
            f.write(f"  Embedding Dim: {config.embedding_dim}\n")
            f.write(f"  Epsilon: {config.epsilon}\n")
            f.write(f"  Privacy Type: {config.privacy_type}\n")
            f.write(f"  Theta: {config.theta}\n\n")
            
            f.write("Attack Model Performance:\n")
            f.write(f"  Accuracy: {attack_results['attack_model_accuracy']:.4f}\n")
            f.write(f"  Features Used: {attack_results['risk_analysis'].get('metadata', {}).get('attack_model_features', 'N/A')}\n")
            f.write(f"  Training Samples: {attack_results['risk_analysis'].get('metadata', {}).get('attack_model_samples', 'N/A')}\n\n")
            
            risk = attack_results['risk_analysis']
            f.write("Privacy Risk Analysis:\n")
            f.write(f"  Risk Score: {risk['risk_score']:.4f}\n")
            f.write(f"  Avg Prediction Error: {risk['avg_prediction_error']:.4f}\n")
            f.write(f"  High Risk Samples: {risk['risk_distribution']['high_risk']} ({risk['risk_distribution']['high_risk']/risk['risk_distribution']['total']*100:.1f}%)\n")
            f.write(f"  Medium Risk Samples: {risk['risk_distribution']['medium_risk']} ({risk['risk_distribution']['medium_risk']/risk['risk_distribution']['total']*100:.1f}%)\n")
            f.write(f"  Low Risk Samples: {risk['risk_distribution']['low_risk']} ({risk['risk_distribution']['low_risk']/risk['risk_distribution']['total']*100:.1f}%)\n\n")
            
            summary = risk['summary']
            f.write("Membership Inference Summary:\n")
            f.write(f"  Total Samples: {summary['total_samples']}\n")
            f.write(f"  Predicted Members: {summary['predicted_members']} ({summary['membership_rate']*100:.1f}%)\n")
            f.write(f"  Predicted Non-Members: {summary['predicted_non_members']}\n")
            f.write(f"  Avg Membership Score: {summary['avg_membership_score']:.4f}\n")
        
        logger.info(f"Inference attack results saved to {attack_file}")
        
        try:
            from evaluator import ModelEvaluator
            evaluator = ModelEvaluator()
            
            plot_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_data = {
                'experiment_id': config.experiment_id,
                'config': config,
                'attack_results': attack_results
            }
            
            plot_path = os.path.join(plot_dir, f'inference_attack_plot_{safe_id}_{timestamp}.png')
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            risk_labels = ['High Risk', 'Medium Risk', 'Low Risk']
            risk_counts = [
                risk['risk_distribution']['high_risk'],
                risk['risk_distribution']['medium_risk'], 
                risk['risk_distribution']['low_risk']
            ]
            axes[0, 0].bar(risk_labels, risk_counts, color=['red', 'orange', 'green'])
            axes[0, 0].set_title('Privacy Risk Distribution')
            axes[0, 0].set_ylabel('Number of Samples')
            
            membership_scores = [r['membership_score'] for r in attack_results['risk_analysis'].get('results', [])]
            axes[0, 1].hist(membership_scores, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Membership Score Distribution')
            axes[0, 1].set_xlabel('Membership Score')
            axes[0, 1].set_ylabel('Frequency')
            
            if hasattr(algorithm, 'loss_history') and algorithm.loss_history:
                axes[1, 0].plot(algorithm.loss_history)
                axes[1, 0].set_title('Training Loss History')
                axes[1, 0].set_xlabel('Round')
                axes[1, 0].set_ylabel('Loss')
            
            attack_metrics = [
                ('Risk Score', risk['risk_score']),
                ('Attack Accuracy', attack_results['attack_model_accuracy']),
                ('Membership Rate', summary['membership_rate']),
                ('Avg Error', risk['avg_prediction_error'])
            ]
            metric_names = [m[0] for m in attack_metrics]
            metric_values = [m[1] for m in attack_metrics]
            axes[1, 1].bar(metric_names, metric_values, alpha=0.7)
            axes[1, 1].set_title('Attack Metrics Summary')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.suptitle(f'Inference Attack Analysis: {config.experiment_id}', fontsize=14)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Inference attack plot saved to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create inference attack plot: {e}")
        
        return attack_file
    
    def _save_validation_results(self, config, eval_results, train_results, output_dir='validation_results'):
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_id = config.experiment_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        val_file = os.path.join(output_dir, f'validation_{safe_id}_{timestamp}.txt')
        
        with open(val_file, 'w') as f:
            f.write(f"Validation Results for Experiment: {config.experiment_id}\n")
            f.write("="*80 + "\n\n")
            
            f.write("Experiment Configuration:\n")
            f.write(f"  Algorithm: {config.algorithm}\n")
            f.write(f"  Setting: {config.setting}\n")
            f.write(f"  Dataset: {config.dataset}\n")
            f.write(f"  Number of Parties: {config.n_parties}\n")
            f.write(f"  Embedding Dimension: {config.embedding_dim}\n")
            f.write(f"  R_max: {config.R_max}\n")
            f.write(f"  T (Rounds): {config.T}\n")
            f.write(f"  T' (Local Steps): {config.T_prime}\n")
            f.write(f"  Learning Rate: {config.learning_rate}\n")
            f.write(f"  Epsilon: {config.epsilon}\n")
            f.write(f"  Delta: {config.delta}\n")
            f.write(f"  Privacy Type: {config.privacy_type}\n")
            f.write(f"  Theta: {config.theta}\n\n")
            
            f.write("Model Evaluation Metrics:\n")
            f.write(f"  Mean Squared Error (MSE): {eval_results.get('mse', 'N/A'):.6f}\n")
            f.write(f"  Root Mean Squared Error (RMSE): {eval_results.get('rmse', 'N/A'):.6f}\n")
            f.write(f"  Mean Absolute Error (MAE): {eval_results.get('mae', 'N/A'):.6f}\n")
            f.write(f"  R² Score: {eval_results.get('r2_score', 'N/A'):.6f}\n")
            f.write(f"  Explained Variance: {eval_results.get('explained_variance', 'N/A'):.6f}\n\n")
            
            f.write("Prediction Statistics:\n")
            f.write(f"  Mean Prediction: {eval_results.get('predictions_mean', 'N/A'):.6f}\n")
            f.write(f"  Std of Predictions: {eval_results.get('predictions_std', 'N/A'):.6f}\n")
            f.write(f"  Mean True Rating: {eval_results.get('true_mean', 'N/A'):.6f}\n")
            f.write(f"  Std of True Ratings: {eval_results.get('true_std', 'N/A'):.6f}\n\n")
            
            if 'communication_rounds' in eval_results:
                f.write("Communication Statistics:\n")
                f.write(f"  Communication Rounds: {eval_results.get('communication_rounds', 'N/A')}\n")
                f.write(f"  Total Bytes: {eval_results.get('total_bytes', 'N/A')}\n")
                f.write(f"  Bytes per Round: {eval_results.get('bytes_per_round', 'N/A')}\n\n")
            
            if train_results and 'final_loss' in train_results and train_results['final_loss'] is not None:
                f.write("Training Results:\n")
                f.write(f"  Final Training Loss: {train_results['final_loss']:.6f}\n")
                if 'loss_history' in train_results and train_results['loss_history']:
                    f.write(f"  Initial Training Loss: {train_results['loss_history'][0] if train_results['loss_history'] else 'N/A':.6f}\n")
                    if len(train_results['loss_history']) > 1:
                        improvement = ((train_results['loss_history'][0] - train_results['final_loss']) / train_results['loss_history'][0]) * 100
                        f.write(f"  Training Loss Improvement: {improvement:.2f}%\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("End of Report\n")
        
        logger.info(f"Validation results saved to {val_file}")
        
        try:
            import matplotlib.pyplot as plt
            plot_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_path = os.path.join(plot_dir, f'validation_plot_{safe_id}_{timestamp}.png')
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            if 'predictions' in eval_results and 'true_values' in eval_results:
                predictions = eval_results['predictions']
                true_values = eval_results['true_values']
                
                axes[0, 0].scatter(true_values, predictions, alpha=0.3, s=10)
                if len(true_values) > 0:
                    min_val = min(min(true_values), min(predictions))
                    max_val = max(max(true_values), max(predictions))
                    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Prediction')
                axes[0, 0].set_xlabel('True Ratings')
                axes[0, 0].set_ylabel('Predicted Ratings')
                axes[0, 0].set_title('True vs Predicted Ratings')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            if train_results and 'loss_history' in train_results and train_results['loss_history']:
                axes[0, 1].plot(train_results['loss_history'])
                axes[0, 1].set_xlabel('Global Rounds')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].set_title('Training Loss History')
                axes[0, 1].grid(True, alpha=0.3)
            
            metrics_to_plot = ['mse', 'rmse', 'mae', 'r2_score']
            metric_values = [eval_results.get(m, 0) for m in metrics_to_plot]
            axes[1, 0].bar(metrics_to_plot, metric_values, alpha=0.7)
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_title('Evaluation Metrics')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            if 'predictions' in eval_results and 'true_values' in eval_results:
                errors = predictions - true_values
                axes[1, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
                axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
                axes[1, 1].set_xlabel('Prediction Error')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Prediction Error Distribution')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'Validation Results: {config.experiment_id}', fontsize=14)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Validation plot saved to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create validation plot: {e}")
        
        return val_file
    
    def save_results(self, output_dir: str = 'results'):
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pkl_filename = os.path.join(output_dir, f'experiment_results_{timestamp}.pkl')
        
        with open(pkl_filename, 'wb') as f:
            pickle.dump(self.results, f)
        
        csv_data = []
        for result in self.results:
            row = {
                'experiment_id': result['experiment_id'],
                'algorithm': result['config'].algorithm,
                'setting': result['config'].setting,
                'dataset': result['config'].dataset,
                'n_parties': result['config'].n_parties,
                'embedding_dim': result['config'].embedding_dim,
                'epsilon': result['config'].epsilon,
                'training_time': result['training_time'],
                'n_users': result['n_users'],
                'n_items': result['n_items']
            }
            
            row.update(result['eval_results'])
            
            if result.get('attack_results'):
                row['attack_accuracy'] = result['attack_results'].get('attack_model_accuracy', 0)
                row['risk_score'] = result['attack_results'].get('risk_analysis', {}).get('risk_score', 0)
                row['avg_membership_score'] = result['attack_results'].get('risk_analysis', {}).get('summary', {}).get('avg_membership_score', 0)
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_filename = os.path.join(output_dir, f'experiment_summary_{timestamp}.csv')
        df.to_csv(csv_filename, index=False)
        
        return pkl_filename, csv_filename
    
    def _run_single_experiment(self, args):
        config, attack_config = args
        try:
            print(f"Running {config.experiment_id}...")
            result = self.run_experiment(config, attack_config)
            return result
        except Exception as e:
            print(f"Failed to run experiment {config.experiment_id}: {e}")
            return None

def run_comparative_experiments():
    print("="*60)
    print("Running comparative experiments with privacy evaluation...")
    print("="*60)
    
    dataset_manager = DatasetManager(data_dir='data')
    experiment_runner = ExperimentRunner(dataset_manager)
    
    attacker_config = AttackerConfig(
        run_attacks=True,
        shadow_models_count=5,
        attack_epochs=15,
        attack_learning_rate=0.02,
        sample_size_for_attack=200
    )
    
    datasets = [ 'MovieLens10M'
                #, 'MovieLens25M', 'MovieLens1M', 
                ]
    algorithms = [
                  'FedADMM', 'FedProx', 'FedAvg',
                  'VFL-SGDMF', 'HFL-SGDMF',
                  'LFL-SGDMF'
                  ]
    n_parties_list = [20, 10, 5, 2, 1]
    privacy_levels = [5.0, 1.0]
    
    all_experiments = []
    
    for dataset in datasets:
        for algorithm in algorithms:
            for n_parties in n_parties_list:
                if algorithm == 'VFL-SGDMF':
                    setting = 'VFL'
                elif algorithm == 'LFL-SGDMF':
                    setting = 'LFL'
                else:
                    setting = 'HFL'
                if setting == 'LFL' and n_parties != 1: #LFL doesn't have any parameter for the number of parties.
                    continue
                for epsilon in privacy_levels:
                    if algorithm.startswith("Fed") and epsilon != 1.0:
                        continue
                    
                    experiment_id = f"{setting}_{algorithm}_{dataset}_{n_parties}_eps{epsilon}"
                    if n_parties == 1:
                        theta = 10
                    elif n_parties >= 2 and n_parties <= 10:
                        theta = 5
                    elif n_parties >= 11:
                        theta = 2
                    else:
                        raise ValueError(f"Invalid number of parties: {n_parties}")
                    config = ExperimentConfig(
                        experiment_id=experiment_id,
                        algorithm=algorithm,
                        setting=setting,
                        dataset=dataset,
                        n_parties=n_parties,
                        embedding_dim=20,
                        T=150,
                        T_prime=5,
                        learning_rate=0.05,
                        epsilon=epsilon,
                        privacy_type='per_rating',
                        theta=theta,
                    )
                    
                    all_experiments.append((config, attacker_config))
    
    print(f"Total experiments to run: {len(all_experiments)}")
    
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel execution")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        future_to_exp = {
            executor.submit(experiment_runner._run_single_experiment, exp): exp 
            for exp in all_experiments
        }
        
        completed = 0
        for future in as_completed(future_to_exp):
            completed += 1
            exp = future_to_exp[future]
            config, _ = exp
            try:
                result = future.result(timeout=3600)
                if result:
                    results.append(result)
                    print(f"✓ Completed {config.experiment_id} ({completed}/{len(all_experiments)})")
                else:
                    print(f"✗ Failed {config.experiment_id} ({completed}/{len(all_experiments)})")
            except Exception as e:
                print(f"✗ Exception for {config.experiment_id}: {e} ({completed}/{len(all_experiments)})")
    
    if results:
        pkl_file, csv_file = experiment_runner.save_results()
        print(f"\nResults saved to {pkl_file} and {csv_file}")
        
        df = pd.read_csv(csv_file)
        print("\nPrivacy-Utility Tradeoff Summary:")
        print(df[['algorithm', 'epsilon', 'rmse', 'risk_score', 'attack_accuracy']].to_string())
    
    return results


if __name__ == "__main__":

    results = run_comparative_experiments()

    
    print("\n" + "="*60)
    print("Experiments completed!")
    print("="*60)