import pandas as pd
import numpy as np
import ast
import os
from scipy.spatial.distance import cosine
from collections import OrderedDict


# Parameters
NUM_LAYERS = 27 # 27 b/c no Layer 0
NUM_EXPERTS = 64
EAMC_CAPACITY = 1000 # How many REAMs to store in EAMC
CACHE_SIZE = 665  # How many experts can fit in GPU cache
WARMUP_COUNT = 800 # How many prompts to include in EAMC
PROMPT_PREDICT_START = 801
PROMPT_PREDICT_END = 801

# Path where CSVs are stored
CSV_FOLDER = '../training_data_eamc'  # Adjust if different

# EAMC Class
class EAMC:
    def __init__(self, capacity):
        self.capacity = capacity
        self.collection = []          # raw REAMs
        self.norm_collection = []     # normalized REAMs, kept in sync

    def add(self, ream):
        norm = normalize_ream(ream)
        if len(self.collection) < self.capacity:
            self.collection.append(ream)
            self.norm_collection.append(norm)
        else:
            self.replace(ream, norm)

    def replace(self, new_ream, new_norm):
        # find the closest stored REAM (by normalized distance)
        dists = [
            cosine(r.flatten(), new_norm.flatten())
            for r in self.norm_collection
        ]
        idx = np.argmin(dists)
        # overwrite both raw and normalized at that index
        self.collection[idx]      = new_ream
        self.norm_collection[idx] = new_norm

# Function to process one CSV file into an rEAM
def process_csv_to_ream(file_path):
    df = pd.read_csv(file_path)
    ream = np.zeros((NUM_LAYERS, NUM_EXPERTS))
    
    for idx, row in df.iterrows():
        layer_id = int(row['Layer ID'])
        expert_list = ast.literal_eval(row['Activated Expert IDs'])  # safely parse string
        
        for expert_id in expert_list:
            ream[layer_id][expert_id] += 1

    return ream

class ExpertCache:
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def access(self, expert_id):
        if expert_id in self.cache:
            # Move accessed item to end (most recently used)
            self.cache.move_to_end(expert_id)
            return True  # Cache hit
        else:
            if len(self.cache) >= self.cache_size:
                # Evict least recently used item (first inserted)
                self.cache.popitem(last=False)
            self.cache[expert_id] = True  # Add new expert
            return False  # Cache miss

def normalize_ream(ream):
    """
    Global max-scale a REAM array.
    ream: np.array of shape (num_layers, num_experts)
    Returns a new array of the same shape with all values in [0,1].
    """
    max_val = ream.max()
    if max_val == 0:
        return ream.copy()   # nothing to scale
    return ream / max_val


def predict_eam(eamc, current_ream, k=5):
    cur_norm = normalize_ream(current_ream).flatten()
    # compute distances only against the cached, normalized collection
    distances = [
        cosine(stored.flatten(), cur_norm)
        for stored in eamc.norm_collection
    ]
    
    # Find indices of k closest neighbors
    k = min(k, len(distances))  # Make sure k is not larger than available REAMs
    closest_indices = np.argsort(distances)[:k]
    
    # Average the k closest REAMs
    avg_ream = np.zeros_like(eamc.collection[0])
    for idx in closest_indices:
        avg_ream += eamc.collection[idx]
    
    avg_ream = avg_ream / k
    return avg_ream

def test_single_csv_predict_mode(file_path, eamc, warmup_count=10, top_k=6):
    df = pd.read_csv(file_path)
    df['token_idx'] = df.groupby('Layer ID').cumcount()
    df = df.sort_values(['token_idx','Layer ID']).reset_index(drop=True)
    df = df.drop(columns='token_idx')

    cache = ExpertCache(CACHE_SIZE)
    hits = misses = 0
    prediction_hits = prediction_misses = 0

    sampled_ieam = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=float)

    warmup_count = 26 * warmup_count

    for idx, row in df.iterrows():
        layer = int(row['Layer ID'])

        experts = ast.literal_eval(row['Activated Expert IDs'])
        for e in experts:
            sampled_ieam[layer, e] += 1
            cache.access((layer, e))   # just warm the cache
        
        if idx >= warmup_count:
            break

    sampled_ream = sampled_ieam.copy()

    for idx, row in df.iterrows():
        if idx < warmup_count:
            continue

        layer = int(row['Layer ID'])
        experts = ast.literal_eval(row['Activated Expert IDs'])

        # a) predict from the current REAM
        predicted_ream = predict_eam(eamc, sampled_ream)

        # b) only consider top_k on this token's layer
        top_experts = np.argsort(predicted_ream[layer])[::-1][:top_k]

        # load predicted experts into cache (don't count these hits/misses)
        for pe in top_experts:
            cache.access((layer, pe))

        # c) now measure prediction hits/misses & cache hits/misses
        for actual in experts:
            if actual in top_experts:
                prediction_hits += 1
            else:
                prediction_misses += 1

            if cache.access((layer, actual)):
                hits += 1
            else:
                misses += 1

        # d) **stream in** the real activations into your REAM
        for e in experts:
            sampled_ream[layer, e] += 1

    return hits, misses, prediction_hits, prediction_misses


def main():
    # 1) Warm-up your EAMC as before
    eamc = EAMC(capacity=EAMC_CAPACITY)
    for n in range(1, WARMUP_COUNT):
        file_path = os.path.join(CSV_FOLDER, f'prompt_{n}_data.csv')
        if os.path.exists(file_path):
            eamc.add(process_csv_to_ream(file_path))
        else:
            print(f"Warning: prompt_{n}_data.csv not found, skipping.")

    print(f"WARMUP COMPLETE: {len(eamc.collection)} rEAMs stored in EAMC.\n")

    # 2) Define which prompts you want to test
    test_prompts = list(range(PROMPT_PREDICT_START, PROMPT_PREDICT_END + 1))

    # 3) Prepare accumulators
    total_hits = total_misses = 0
    total_pred_hits = total_pred_misses = 0
    f1_scores = []

    # 4) Loop over each test file
    for test_n in test_prompts:
        test_file = f'prompt_{test_n}_data.csv'
        test_path = os.path.join(CSV_FOLDER, test_file)
        if not os.path.exists(test_path):
            print(f"  • {test_file} not found, skipping.")
            continue

        hits, misses, pred_hits, pred_misses = \
            test_single_csv_predict_mode(test_path, eamc, warmup_count=10)

        total_hits       += hits
        total_misses     += misses
        total_pred_hits  += pred_hits
        total_pred_misses+= pred_misses

        if (hits+misses) > 0:
            hit_rate = hits / (hits+misses)
        else:
            hit_rate = 0.0
        if (pred_hits+pred_misses) > 0:
            pred_rate = pred_hits / (pred_hits+pred_misses)
        else:
            pred_rate = 0.0

        print(f"Results for {test_file}:  cache hit rate {hit_rate:.2%}, "
              f"pred hit rate {pred_rate:.2%}")

    # 5) Compute aggregated metrics
    overall_cache_rate = total_hits / (total_hits + total_misses)
    overall_pred_rate  = total_pred_hits / (total_pred_hits + total_pred_misses)

    print("\n=== AGGREGATED OVER ALL TESTS ===")
    print(f"Cache hit rate      : {overall_cache_rate:.2%}")
    print(f"Prediction hit rate : {overall_pred_rate:.2%}")

if __name__ == "__main__":
    main()
