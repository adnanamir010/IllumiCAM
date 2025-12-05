import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import argparse
import glob
from tqdm import tqdm

def load_data(data_root):
    """Load .wp files and extract chromaticity."""
    data_list = []
    print(f"Loading .wp files from {data_root}...")
    
    search_pattern = os.path.join(data_root, "**", "*.wp")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"No .wp files found in {search_pattern}")
        return pd.DataFrame()

    for wp_file in tqdm(files, desc="Reading files"):
        try:
            with open(wp_file, "r") as f:
                line = f.read().strip()
                values = line.replace("\t", " ").split()
                if len(values) >= 3:
                    r, g, b = float(values[0]), float(values[1]), float(values[2])
                    total = r + g + b
                    if total > 0:
                        data_list.append({
                            'mean_r': r/total,
                            'mean_g': g/total,
                            'mean_b': b/total,
                        })
        except Exception as e:
            print(f"Error reading {wp_file}: {e}")

    return pd.DataFrame(data_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="Data/Nikon_D810", help="Path to raw data")
    parser.add_argument("--output", type=str, default="centroids.npy", help="Output file")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters")
    args = parser.parse_args()

    df = load_data(args.data_root)
    if df.empty:
        print("No data found.")
        return

    print(f"performing KMeans clustering with k={args.k}...")
    X = df[['mean_r', 'mean_g', 'mean_b']].values
    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # Sort clusters by coolness (B/R ratio or similar) to match labels
    # This logic mimics the notebook's sorting
    sorted_indices = sorted(range(args.k), key=lambda i: kmeans.cluster_centers_[i][2] / kmeans.cluster_centers_[i][0])
    
    label_template = ["Very_Warm", "Warm", "Neutral", "Cool", "Very_Cool"]
    
    centroids_dict = {}
    for rank, cluster_id in enumerate(sorted_indices):
        name = label_template[rank] if rank < len(label_template) else f"Cluster_{rank}"
        centroids_dict[name] = kmeans.cluster_centers_[cluster_id]
        
    np.save(args.output, centroids_dict)
    print(f"Saved centroids to {args.output}")
    print(centroids_dict)

if __name__ == "__main__":
    main()
