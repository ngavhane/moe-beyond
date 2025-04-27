import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

class PredictionMetrics:
    def __init__(self):
        self.total = 0
        self.true_pos = 0
        self.false_pos = 0
        self.false_neg = 0
        
    def update(self, predicted, actual):
        pred_set = set(predicted)
        actual_set = set(actual)
        
        self.true_pos += len(pred_set & actual_set)
        self.false_pos += len(pred_set - actual_set)
        self.false_neg += len(actual_set - pred_set)
        self.total += 1
        
    @property
    def precision(self):
        return self.true_pos / (self.true_pos + self.false_pos) if self.total else 0
    
    @property
    def recall(self):
        return self.true_pos / (self.true_pos + self.false_neg) if self.total else 0
    
    @property
    def f1(self):
        p = self.precision
        r = self.recall
        return 2*(p*r)/(p+r) if (p+r) else 0

class IterationEAM:
    def __init__(self, num_layers, num_experts):
        self.matrix = np.zeros((num_layers, num_experts))
        self.actual_activations = defaultdict(set)  # Track actual used experts
    
    def record_actual(self, layer_idx, expert_idx):
        self.actual_activations[layer_idx].add(expert_idx)
        self.matrix[layer_idx][expert_idx] += 1

class EAMCluster:
    def __init__(self, num_layers, num_experts, max_clusters=30):
        self.eamc = []
        self.kmeans = KMeans(n_clusters=max_clusters)
        self.vectorized = []
        self.num_layers = num_layers
        self.num_experts = num_experts
        
    def add_request(self, rEAM):
        vec = rEAM.matrix.flatten()
        self.eamc.append(rEAM)
        self.vectorized.append(vec)
        
        if len(self.vectorized) % 100 == 0:
            self.kmeans.fit(self.vectorized)
    
    def predict_experts(self, iEAM, top_k=2):
        query_vec = np.array(iEAM.matrix).flatten()
        if len(self.vectorized) == 0:
            return defaultdict(set)
            
        cluster_idx = self.kmeans.predict([query_vec])[0]
        cluster_center = self.kmeans.cluster_centers_[cluster_idx]
        
        # Reshape cluster center to EAM format
        predicted_eam = cluster_center.reshape(
            (self.num_layers, self.num_experts))
        
        # Get top-k experts per layer
        predictions = defaultdict(set)
        for layer in range(self.num_layers):
            experts = predicted_eam[layer].argsort()[-top_k:]
            predictions[layer].update(experts)
            
        return predictions

def simulate_moe_inference(num_requests=100, num_layers=8, num_experts=8, top_k=2):
    metrics = PredictionMetrics()
    cluster = EAMCluster(num_layers, num_experts)
    
    for _ in range(num_requests):
        rEAM = IterationEAM(num_layers, num_experts)
        
        # Simulate multiple iterations per request
        for _ in range(20):  # 20 tokens per request
            iEAM = IterationEAM(num_layers, num_experts)
            
            # Simulate actual expert selection (replace with real router logic)
            for layer in range(num_layers):
                # Randomly select actual experts (simulated routing)
                actual_experts = np.random.choice(
                    num_experts, size=top_k, replace=False)
                for expert in actual_experts:
                    iEAM.record_actual(layer, expert)
            
            # Get predictions
            predicted = cluster.predict_experts(iEAM, top_k)
            
            # Compare predictions vs actuals
            for layer in range(num_layers):
                actual = iEAM.actual_activations.get(layer, set())
                pred = predicted.get(layer, set())
                metrics.update(pred, actual)
            
            # Accumulate into request EAM
            rEAM.matrix += iEAM.matrix
        
        # Add completed request to cluster
        cluster.add_request(rEAM)
    
    print(f"Precision: {metrics.precision:.2%}")
    print(f"Recall: {metrics.recall:.2%}") 
    print(f"F1 Score: {metrics.f1:.2%}")
    print(f"Total Comparisons: {metrics.total}")

# Example usage
simulate_moe_inference(num_requests=100, num_layers=8, num_experts=8, top_k=2)
