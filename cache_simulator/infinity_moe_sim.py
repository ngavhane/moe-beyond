import pandas as pd
import numpy as np
import ast
import os
from scipy.spatial.distance import cosine
from collections import OrderedDict


# Parameters
NUM_LAYERS = 27 # 27 b/c no Layer 0
NUM_EXPERTS = 64
EAMC_CAPACITY = 1000
CACHE_SIZE = 15  # How many experts can fit in GPU cache
WARMUP_COUNT = 150
PROMPT_PREDICTION = 501

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


def predict_eam(eamc, current_ream):
    cur_norm = normalize_ream(current_ream).flatten()
    # compute distances only against the cached, normalized collection
    distances = [
        cosine(stored.flatten(), cur_norm)
        for stored in eamc.norm_collection
    ]
    best_idx = np.argmin(distances)
    return eamc.collection[best_idx]

def test_single_csv_predict_mode(file_path, eamc, warmup_count=10, top_k=6):
    # 1) load & sort so that cumcount gives 0..n per layer in token order
    df = pd.read_csv(file_path)
    df = df.sort_values(['Layer ID', 'Batch Number']).reset_index(drop=True)

    cache = ExpertCache(CACHE_SIZE)
    hits = misses = 0
    prediction_hits = prediction_misses = 0

    # identify warm-up rows: first warmup_count rows in each Layer ID
    is_warmup = df.groupby('Layer ID').cumcount() < warmup_count

    # 2) Warm-up: build sampled IEAM and pre-fill cache (no metrics yet)
    sampled_ieam = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=float)
    for _, row in df[is_warmup].iterrows():
        layer = int(row['Layer ID'])
        experts = ast.literal_eval(row['Activated Expert IDs'])
        for e in experts:
            sampled_ieam[layer, e] += 1
            cache.access((layer, e))   # just warm the cache

    # we'll mutate this as we go
    sampled_ream = sampled_ieam.copy()

    # 3) Evaluation: for each subsequent token
    for _, row in df[~is_warmup].iterrows():
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
    # Initialize EAMC
    eamc = EAMC(capacity=EAMC_CAPACITY)

    # Warmup Loop: Process all CSVs
    for n in range(1, WARMUP_COUNT):
        file_name = f'prompt_{n}_data.csv'
        file_path = os.path.join(CSV_FOLDER, file_name)
        
        if os.path.exists(file_path):
            ream = process_csv_to_ream(file_path)
            eamc.add(ream)
        else:
            print(f"Warning: {file_name} not found.")

    print(f"WARMUP COMPLETE: {len(eamc.collection)} rEAMs stored in EAMC.")

    test_n = PROMPT_PREDICTION
    test_file = f'prompt_{test_n}_data.csv'
    test_path = os.path.join(CSV_FOLDER, test_file)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    hits, misses, pred_hits, pred_misses = test_single_csv_predict_mode(
        test_path, eamc, warmup_count=10
    )

    total_accesses = hits + misses
    total_preds    = pred_hits + pred_misses

    print(f"TEST RESULTS on {test_file}:")
    print(f"  Cache hits   : {hits} / {total_accesses} ({hits/total_accesses:.2%})")
    print(f"  Cache misses : {misses}")
    print(f"  Pred hits    : {pred_hits} / {total_preds} ({pred_hits/total_preds:.2%})")
    print(f"  Pred misses  : {pred_misses}")


if __name__ == "__main__":
    main()
