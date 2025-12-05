import os
import numpy as np
import argparse
from tqdm import tqdm
from illuminant_estimation.data.lsmi_dataset import map_illuminant_to_cluster

def load_cluster_centers(path):
    """
    Load cluster centers from .npy file.
    Handles the dictionary format found in cluster_centers.npy
    """
    data = np.load(path, allow_pickle=True)
    if data.shape == ():
        data = data.item()
    
    # Ensure we have the 5 expected keys in order
    keys = ['Very_Warm', 'Warm', 'Neutral', 'Cool', 'Very_Cool']
    centers = []
    for k in keys:
        if k in data:
            centers.append(data[k])
        else:
            print(f"Warning: Key {k} not found in cluster centers.")
            # Fallback or error
            
    return np.array(centers), keys

def main():
    parser = argparse.ArgumentParser(description="Prepare LSMI Masks for WSSS")
    parser.add_argument("--lsmi_root", type=str, default="Data/LSMI", help="Path to LSMI dataset")
    parser.add_argument("--output_dir", type=str, default="Data/LSMI_Masks", help="Output directory for masks")
    parser.add_argument("--centroids_file", type=str, default="cluster_centers.npy", help="Path to centroids file")
    args = parser.parse_args()

    if not os.path.exists(args.lsmi_root):
        print(f"LSMI root {args.lsmi_root} does not exist. Please ensure dataset is placed there.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Load centroids
    centers, center_names = load_cluster_centers(args.centroids_file)
    print(f"Loaded {len(centers)} cluster centers: {center_names}")

    # Placeholder for LSMI iteration
    # In reality, we would iterate over the dataset, load GT, and generate masks
    print("Starting mask generation...")
    
    # Example logic (commented out until data is present):
    # for image_name in tqdm(os.listdir(args.lsmi_root)):
    #     if not image_name.endswith(".jpg"): continue
    #     
    #     # Load GT illuminants for this image (e.g. from a .mat file)
    #     # gt_illuminants = load_gt(image_name) 
    #     
    #     # Map each illuminant to a cluster
    #     # ...
    #     
    #     # Save mask
    #     # np.save(os.path.join(args.output_dir, image_name.replace(".jpg", ".npy")), mask)

    print("Mask generation logic is ready. Waiting for data integration.")

if __name__ == "__main__":
    main()
