import pandas as pd
import numpy as np
import ast
import os
from scipy.spatial.distance import cosine
from collections import OrderedDict
from sklearn.cluster import KMeans


# Parameters
NUM_LAYERS = 27 # 27 b/c no Layer 0
NUM_EXPERTS = 64
EAMC_CAPACITY = 1000 # How many REAMs to store in EAMC
EAMC_SIZE = 10 # How many prompts to include in EAMC
PROMPT_PREDICT_START = 1
PROMPT_PREDICT_END = 100
NUM_CLUSTERS = 1  # Number of clusters for k-means

# Path where CSVs are stored
CSV_FOLDER_EAMC = '../training_data_eamc'  # Adjust if different
CSV_FOLDER = '../test_datasets/predicted_csvs'  # Adjust if different

# EAMC Class
class EAMC:
    def __init__(self, capacity):
        self.capacity = capacity
        self.collection = []          # raw REAMs
        self.cluster_centroids = None  # Will hold cluster centroids
        self.cluster_assignments = None  # Will hold cluster assignments

    def add(self, ream):
        if len(self.collection) < self.capacity:
            self.collection.append(ream)
        else:
            self.replace(ream)

    def replace(self, new_ream):
        # find the closest stored REAM (by distance)
        dists = [
            cosine(r.flatten(), new_ream.flatten())
            for r in self.collection
        ]
        idx = np.argmin(dists)
        self.collection[idx]      = new_ream

    def cluster_reams(self, n_clusters=NUM_CLUSTERS):
        """Perform k-means clustering on collected REAMs"""
        if len(self.collection) < n_clusters:
            print(f"Warning: Not enough REAMs ({len(self.collection)}) for {n_clusters} clusters")
            n_clusters = max(1, len(self.collection) // 2)
        
        # Flatten REAMs for clustering
        flat_reams = np.array([ream.flatten() for ream in self.collection])
        
        # Perform k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_assignments = kmeans.fit_predict(flat_reams)
        
        # Store centroids in original REAM shape
        self.cluster_centroids = []
        for centroid in kmeans.cluster_centers_:
            self.cluster_centroids.append(centroid.reshape(NUM_LAYERS, NUM_EXPERTS))
            
        print(f"K-means clustering complete: {n_clusters} clusters created")


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

        # Randomly initialize cache (key: (layer, expert), value: bool)
        all_experts = [(layer, expert) for layer in range(1, NUM_LAYERS) for expert in range(NUM_EXPERTS)]
        np.random.shuffle(all_experts)  # Shuffle all possible experts
        for expert_id in all_experts[:cache_size]:
            self.cache[expert_id] = True

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


def predict_eam(eamc, current_ream, k=5):
    curr_ream = current_ream.flatten()
    
    if eamc.cluster_centroids is not None:
        # Find the closest cluster centroid
        similarities = [
            1 - cosine(centroid.flatten(), curr_ream)
            for centroid in eamc.cluster_centroids
        ]
        
        # Use the most similar centroid
        best_idx = np.argmax(similarities)
        
        # Return the centroid as the prediction
        return eamc.cluster_centroids[best_idx]
    else:
        # Fall back to original method if clustering wasn't performed
        similarities = [
            1 - cosine(stored.flatten(), curr_ream)
            for stored in eamc.collection
        ]
        
        # Find indices of k most similar neighbors (highest similarity)
        k = min(k, len(similarities))  # Make sure k is not larger than available REAMs
        closest_indices = np.argsort(similarities)[::-1][:k]  # Descending order for similarity
    
        # Print cosine similarities of closest neighbors
        for i in range(closest_indices.shape[0]):
            print(similarities[closest_indices[i]])
        
        # Average the k closest REAMs
        avg_ream = np.zeros_like(eamc.collection[0])
        for idx in closest_indices:
            avg_ream += eamc.collection[idx]
        
        avg_ream = avg_ream / k
        return avg_ream

def test_single_csv_predict_mode(file_path, eamc, cache_size, warmup_count=10, top_k=6):
    df = pd.read_csv(file_path)
    df['token_idx'] = df.groupby('Layer ID').cumcount()
    df = df.sort_values(['token_idx','Layer ID']).reset_index(drop=True)
    df = df.drop(columns='token_idx')

    cache = ExpertCache(cache_size)
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

def test_single_csv_predict_mode_with_predicted(file_path, cache_size, warmup_count=10, top_k=6):
    df = pd.read_csv(file_path)
    df['token_idx'] = df.groupby('Layer ID').cumcount()
    df = df.sort_values(['token_idx','Layer ID']).reset_index(drop=True)
    df = df.drop(columns='token_idx')

    cache = ExpertCache(cache_size)
    hits = misses = 0
    prediction_hits = prediction_misses = 0

    
    warmup_count = 26 * warmup_count

    for idx, row in df.iterrows():
        layer = int(row['Layer ID'])

        experts = ast.literal_eval(row['Activated Expert IDs'])
        for e in experts:
            cache.access((layer, e))   # just warm the cache
        
        if idx >= warmup_count:
            break

    for idx, row in df.iterrows():
            if idx < warmup_count:
                continue

            layer = int(row['Layer ID'])
            experts = ast.literal_eval(row['Activated Expert IDs'])

            top_experts = ast.literal_eval(row['Predicted Expert IDs'])

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


    return hits, misses, prediction_hits, prediction_misses 

def main():
    # 1) Warm-up your EAMC as before
    eamc = EAMC(capacity=EAMC_CAPACITY)
    for n in range(1, EAMC_SIZE):
        file_path = os.path.join(CSV_FOLDER_EAMC, f'prompt_{n}_data.csv')
        if os.path.exists(file_path):
            eamc.add(process_csv_to_ream(file_path))
        else:
            print(f"Warning: prompt_{n}_data.csv not found, skipping.")

    print(f"WARMUP COMPLETE: {len(eamc.collection)} rEAMs stored in EAMC.\n")
    
    # Perform clustering on collected REAMs
    eamc.cluster_reams(n_clusters=NUM_CLUSTERS)

    # 2) Define which prompts you want to test
    test_prompts = list(range(PROMPT_PREDICT_START, PROMPT_PREDICT_END + 1))

    # Test different cache sizes (10% increments of 1664)
    cache_sizes = [int(1664 * (i/10)) for i in range(1, 11)]
    results = []

    for cache_size in cache_sizes:
        # 3) Prepare accumulators
        total_hits = total_misses = 0
        total_pred_hits = total_pred_misses = 0

        # 4) Loop over each test file
        for test_n in test_prompts:
            test_file = f'prompt_{test_n}_data_predicted.csv'
            test_path = os.path.join(CSV_FOLDER, test_file)
            if not os.path.exists(test_path):
                print(f"  • {test_file} not found, skipping.")
                continue

            hits, misses, pred_hits, pred_misses = \
                test_single_csv_predict_mode_with_predicted(test_path, cache_size, warmup_count=10)

            # hits, misses, pred_hits, pred_misses = \
            #     test_single_csv_predict_mode(test_path, eamc, cache_size, warmup_count=10)

            if hits == 0 and misses == 0:
                continue
            if pred_hits == 0 and pred_misses == 0:
                continue

            total_hits       += hits
            total_misses     += misses
            total_pred_hits  += pred_hits
            total_pred_misses+= pred_misses

        # 5) Compute aggregated metrics
        overall_cache_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
        overall_pred_rate  = total_pred_hits / (total_pred_hits + total_pred_misses) if (total_pred_hits + total_pred_misses) > 0 else 0

        results.append({
            'cache_size': cache_size,
            'cache_hit_rate': overall_cache_rate,
            'prediction_hit_rate': overall_pred_rate
        })

    # Print results in a table format
    print("\n=== RESULTS FOR DIFFERENT CACHE SIZES ===")
    print(f"{'Cache Size':<12} {'Cache Hit Rate':<20} {'Prediction Hit Rate':<20}")
    print("-" * 55)
    for result in results:
        print(f"{result['cache_size']:<12} {result['cache_hit_rate']:.2%}{'':<12} {result['prediction_hit_rate']:.2%}")

if __name__ == "__main__":
    main()
