"""
Test VFL training with noise disabled to verify loss decreases properly
"""
import os
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from sgdmf_vfl import VFL_SGDMF
from party import Party
from differential_privacy import DifferentialPrivacy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load small sample of MovieLens1M
df = pd.read_csv('data/MovieLens1M/ratings.csv', nrows=1000)
df['userId'] = df['userId'] - 1
df['movieId'] = df['movieId'] - 1

print(f"Loaded {len(df)} ratings")
print(f"Users: {df['userId'].max() + 1}, Items: {df['movieId'].max() + 1}")

# Split data between 2 parties (vertical split by items)
n_users = int(df['userId'].max()) + 1
n_items = int(df['movieId'].max()) + 1

items_per_party = (n_items + 1) // 2
party1_items = set(range(items_per_party))
party2_items = set(range(items_per_party, n_items))

party1_data = df[df['movieId'].isin(party1_items)][['userId', 'movieId', 'rating']].values
party2_data = df[df['movieId'].isin(party2_items)][['userId', 'movieId', 'rating']].values

print(f"\nParty 1 data shape: {party1_data.shape}")
print(f"Party 2 data shape: {party2_data.shape}")

# Initialize VFL
embedding_dim = 8
R_max = 50.0
dp = DifferentialPrivacy(epsilon=1e6, delta=1e-6, R_max=R_max)

vfl = VFL_SGDMF(
    n_users=n_users,
    n_items=n_items,
    embedding_dim=embedding_dim,
    R_max=R_max,
    epsilon=1e6,
    delta=1e-6,
    privacy_type='per_user',
    party_data_list=[party1_data, party2_data],
)

print(f"\nInitialized VFL with {vfl.n_parties} parties")

# Train WITH noise disabled
print("\n" + "="*60)
print("Training VFL with NOISE DISABLED")
print("="*60)

vfl.loss_history = []
vfl.train(
    [party1_data, party2_data],
    T=5,  # 5 rounds
    T_prime=2,  # 2 local steps per round
    learning_rate=0.01,
    sampling_rate=0.1,
    add_noise=False  # DISABLE NOISE
)

print("\nFinal loss history (NO NOISE):")
for t, loss in enumerate(vfl.loss_history):
    print(f"  Round {t}: {loss:.4f}")

if vfl.loss_history:
    initial_loss = vfl.loss_history[0]
    final_loss = vfl.loss_history[-1]
    improvement = initial_loss - final_loss
    pct_improvement = 100 * improvement / initial_loss if initial_loss > 0 else 0
    print(f"\nLoss reduction: {initial_loss:.4f} → {final_loss:.4f}")
    print(f"Improvement: {improvement:.4f} ({pct_improvement:.1f}%)")
    
    if final_loss < initial_loss:
        print("✓ SUCCESS: Loss decreased without noise!")
    else:
        print("✗ FAILED: Loss still increased without noise - other issues present")

# Now test WITH noise to show the difference
print("\n" + "="*60)
print("Training VFL with NOISE ENABLED for comparison")
print("="*60)

# Reinitialize for clean comparison
vfl2 = VFL_SGDMF(
    n_users=n_users,
    n_items=n_items,
    embedding_dim=embedding_dim,
    R_max=R_max,
    epsilon=1e6,
    delta=1e-6,
    privacy_type='per_user',
    party_data_list=[party1_data, party2_data],
)

vfl2.loss_history = []
vfl2.train(
    [party1_data, party2_data],
    T=5,
    T_prime=2,
    learning_rate=0.01,
    sampling_rate=0.1,
    add_noise=True  # ENABLE NOISE
)

print("\nFinal loss history (WITH NOISE):")
for t, loss in enumerate(vfl2.loss_history):
    print(f"  Round {t}: {loss:.4f}")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"Initial loss (no noise):   {vfl.loss_history[0]:.4f}")
print(f"Final loss (no noise):     {vfl.loss_history[-1]:.4f}")
print(f"Initial loss (with noise): {vfl2.loss_history[0]:.4f}")
print(f"Final loss (with noise):   {vfl2.loss_history[-1]:.4f}")
