import numpy as np

class DataPartitioner:
    @staticmethod
    def partition_vertical(ratings, n_parties, n_items, split_by='random', categories=None):
        item_indices = list(range(n_items))
        if(split_by == 'category' and categories):
            party_items = DataPartitioner._split_by_category(item_indices, categories, n_parties)
        elif (split_by == 'balanced'):
            party_items = DataPartitioner._split_balanced(item_indices, n_parties)
        else:
            np.random.shuffle(item_indices)
            party_items = np.array_split(item_indices, n_parties)
        party_data = [[] for _ in range(n_parties)]
        
        item_to_party = {}
        for party_index, items in enumerate(party_items):
            for item_index in items:
                item_to_party[item_index] = party_index
        
        for user_index, item_index, rating in ratings:
            if(item_index in item_to_party):
                party_index = item_to_party[item_index]
                party_data[party_index].append((user_index, item_index, rating))
                
        party_maps = []
        
        for party_index, items in enumerate(party_items):
            party_map = {
                'global_inidices': items.tolist() if hasattr(items, 'tolist') else list(items),
                'global_to_local': {item: index for index,item in enumerate(items)},
                'local_to_global': {index: item for index, item in enumerate(items)}
            }
            party_maps.append(party_map)
        return party_data, party_maps
    
    @staticmethod
    def partition_horizontal(ratings, n_parties, n_users, split_by='random'):
        user_indices = list(range(n_users))
        if(split_by == 'balanced'):
            party_users = DataPartitioner._split_balanced(user_indices, n_parties)
        else:
            np.random.shuffle(user_indices)
            party_users = np.array_split(user_indices, n_parties)
            
        party_data = [[] for _ in range(n_parties)]
        
        user_to_party = {}
        for party_index, users in enumerate(party_users):
            for user_index in users:
                user_to_party[user_index] = party_index
                
        for user_index, item_index, rating in ratings:
            if(user_index in user_to_party):
                party_index = user_to_party[user_index]
                party_data[party_index].append((user_index, item_index, rating))
                
        party_maps = []
        for party_index, users in enumerate(party_users):
            party_map = {
                'global_indices': users.tolist() if hasattr(users, 'tolist') else list(users),
                'global_to_local': {user: index for index, user in enumerate(users)},
                'local_to_global': {index: user for index, user in enumerate(users)}
            }
            party_maps.append(party_map)
            
        return party_data, party_maps

    @staticmethod
    def partition_local(ratings, num_users):
        party_data = [[] for _ in range(num_users)]
        for user_index, item_index, rating in ratings:
            if (user_index < num_users):
                party_data[user_index].append((user_index, item_index, rating))
                
        party_maps = []
        for user_index in range(num_users):
            party_map = {
                'global_indices': [user_index],
                'global_to_local': {user_index: 0},
                'local_to_global': {0: user_index}
            }
            party_maps.append(party_map)
            
        return party_data, party_maps
        
    @staticmethod
    def _split_balanced(indices, num_parties):
        np.random.shuffle(indices)
        splits = np.array_split(indices,num_parties)
        
        sizes = [len(split) for split in splits]
        min_size = min(sizes)
        max_size = max(sizes)
        
        if max_size - min_size > 1:
            return DataPartitioner._balance_splits(splits)
        return splits
    
    @staticmethod
    def _balance_splits(splits):
        sizes = [len(split) for split in splits]
        total_items = sum(sizes)
        target_size = total_items // len(splits)
        balanced = []
        extra_items = []
        for split in splits:
            if len(split) > target_size:
                balanced.append(split[:target_size])
                extra_items.extend(split[target_size:])
            else:
                balanced.append(split)
        
        while extra_items:
            for i in range(len(balanced)):
                if len(balanced[i]) < target_size and extra_items:
                    balanced[i] = np.append(balanced[i], extra_items.pop(0))
        
        return balanced

    @staticmethod
    def _split_by_category(item_indices, categories, num_parties):
        if categories is None or len(categories) != len(item_indices):
            np.random.shuffle(item_indices)
            return np.array_split(item_indices, num_parties)
        
        category_to_items = {}
        for item_idx, category in enumerate(categories):
            if category not in category_to_items:
                category_to_items[category] = []
            category_to_items[category].append(item_idx)
        
        category_counts = {cat: len(items) for cat, items in category_to_items.items()}
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1])
        
        party_assignments = [[] for _ in range(num_parties)]
        party_sizes = [0] * num_parties
        
        for category, items in sorted_categories:
            target_party = np.argmin(party_sizes)
            party_assignments[target_party].extend(items)
            party_sizes[target_party] += len(items)
        
        party_items = []
        for assignment in party_assignments:
            if assignment:
                party_items.append(np.array(assignment))
            else:
                party_items.append(np.array([]))
        
        return party_items
    
    @staticmethod
    def create_train_test_split(ratings, test_ratio=0.2, random_seed=42):
        np.random.seed(random_seed)
        np.random.shuffle(ratings)
        
        split_idx = int(len(ratings) * (1 - test_ratio))
        train_ratings = ratings[:split_idx]
        test_ratings = ratings[split_idx:]
        
        return train_ratings, test_ratings
    
    @staticmethod
    def create_cross_validation_folds(ratings, n_folds=5, random_seed=42):
        np.random.seed(random_seed)
        np.random.shuffle(ratings)
        
        fold_size = len(ratings) // n_folds
        folds = []
        
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(ratings)
            test_ratings = ratings[start_idx:end_idx]
            train_ratings = ratings[:start_idx] + ratings[end_idx:]
            folds.append((train_ratings, test_ratings))
        
        return folds