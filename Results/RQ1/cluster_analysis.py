import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# === Load Data ===
def load_active_users_data(dataset_file='Data/active_users_dataset.json'):
    with open(dataset_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# === Classify Users by Stance ===
def classify_users_by_stance(dataset, uncertainty_lower=0.8, uncertainty_upper=1.0):
    user_classifications = {}
    for username, comments in dataset.items():
        stances = [c['stance'] for c in comments if c['stance'] is not None]
        if len(stances) == 0:
            continue
        mean_stance = np.mean(stances)
        if mean_stance < uncertainty_lower:
            classification = 'right'
        elif mean_stance > uncertainty_upper:
            classification = 'left'
        else:
            classification = 'uncertain'
        user_classifications[username] = classification
    return user_classifications

# === Extract User Features ===
def prepare_user_features(dataset, user_classifications, min_comments=10):
    user_features = {}
    for username, comments in dataset.items():
        if username not in user_classifications:
            continue
        channel_counts = Counter()
        for comment in comments:
            channel = comment.get('ChannelLeaning', 'Unknown')
            if channel in ['Left', 'Right', 'Center']:
                channel_counts[channel] += 1
        total = sum(channel_counts.values())
        if total >= min_comments:
            proportions = {c: channel_counts[c] / total for c in ['Left', 'Right', 'Center']}
            user_features[username] = {
                'stance': user_classifications[username],
                'features': [proportions.get('Left', 0), proportions.get('Right', 0), proportions.get('Center', 0)]
            }
    return user_features

# === Clustering Function ===
def cluster_users_by_group(user_features, stance_group, max_k=6):
    users = [u for u, d in user_features.items() if d['stance'] == stance_group]
    if len(users) < 10:
        print(f"Not enough {stance_group}-users for clustering (n={len(users)}). Skipping.")
        return None
    X = np.array([user_features[u]['features'] for u in users])
    silhouette_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append((k, silhouette_score(X, labels)))
    best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(X)
    cluster_assignments = {users[i]: int(final_labels[i]) for i in range(len(users))}
    print(f"{stance_group.title()} group: best k={best_k}, silhouette={best_score:.3f}")
    return {
        'stance_group': stance_group,
        'best_k': best_k,
        'silhouette': best_score,
        'assignments': cluster_assignments,
        'features': X.tolist(),
        'users': users
    }

# === Master Analysis ===
def run_stance_based_clustering(dataset_file='Data/active_users_dataset.json'):
    dataset = load_active_users_data(dataset_file)
    user_classifications = classify_users_by_stance(dataset)
    user_features = prepare_user_features(dataset, user_classifications)
    results = {}
    for stance_group in ['left', 'right', 'uncertain']:
        results[stance_group] = cluster_users_by_group(user_features, stance_group)
    # Save results
    with open('stance_based_clustering_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Results saved to stance_based_clustering_results.json")
    return results

# Run if script is main
if __name__ == "__main__":
    results = run_stance_based_clustering()
