Here is a comprehensive plan to professionalize your Part 1 codebase and immediately execute the first steps of Part 2.

### **Phase 1: Refactoring Part 1 into a Python Module**

To move away from the notebook structure, you should organize your code into a modular package. This separates concerns and makes the model easy to import for Part 2.

**Proposed Directory Structure:**

```text
illuminant_estimation/
├── data/
│   ├── __init__.py
│   ├── dataset.py       # CustomDataset class, data loading logic
│   └── augmentation.py  # augment_image_crop_resize, create_augmented_image
├── models/
│   ├── __init__.py
│   └── cnn.py           # IlluminantCNN class
├── utils/
│   ├── __init__.py
│   ├── metrics.py       # Angular error, plotting functions
│   └── image.py         # tensor_to_rgb, normalization helpers
├── train.py             # Main training loop, arguments parsing
└── export_centroids.py  # One-time script to save K-Means centroids for Part 2
```

#### **1. `models/cnn.py`**

Move the `IlluminantCNN` class here.

  * **Action:** Ensure you import `torch` and `torch.nn`.
  * **Refinement:** Add a `num_classes` parameter to `__init__` so it defaults to 5 but is flexible.

#### **2. `data/dataset.py` & `augmentation.py`**

  * **`augmentation.py`**: Extract `augment_image_crop_resize` and `create_augmented_image`.
  * **`dataset.py`**: Include the `CustomDataset` class.
      * **Crucial Change:** The notebook currently relies on global variables for `transform`. Move the transformation logic inside the `__getitem__` or pass `transforms` as an argument to `__init__`.

#### **3. `train.py`**

This script should handle the "Training" and "Validation" sections of your notebook.

  * **Input:** Should accept arguments like `--data_dir`, `--epochs`, `--batch_size`.
  * **Logic:** Setup `DataLoader`, initialize `IlluminantCNN`, run the training loop (Adam, CrossEntropy, ReduceLROnPlateau), and save `best_illuminant_cnn.pth`.

#### **4. `export_centroids.py` (Critical for Part 2)**

You specifically excluded EDA, but Part 2 **requires** the cluster centers (the specific RGB values that define "Warm", "Cool", etc.) to map the new LSMI data.

  * **Task:** Write a script that loads your training data, runs the K-Means (k=5) exactly as the notebook did, and saves the 5 cluster centroids to a file (e.g., `centroids.npy` or `centroids.json`).

-----

### **Phase 2: Execution Plan**

According to your request and the research report, here is how to proceed with the Multi-Illuminant (LSMI) phase.

#### **Step 1: Baseline Evaluation (Zero-Shot on LSMI)**

Before building complex pipelines, you need to see how your single-illuminant model reacts to multi-illuminant scenes. [cite_start]The research report hypothesizes that it will produce "sparse, dot-like activations"[cite: 20].

**Action Plan:**

1.  **Load Data:** Load 10-20 images from the **LSMI dataset** (raw input images).
2.  **Run Inference:** Pass them through your trained `IlluminantCNN` (from Part 1).
      * *Note:* The model will output a single class (e.g., "Cool"). This will likely correspond to the dominant light source in the scene.
3.  **Generate Heatmaps (Benchmark):** Use the **Grad-CAM** code already present in your notebook (refactored into `utils/visualization.py`).
      * **Target:** Generate heatmaps for the top 2 predicted classes.
4.  **Observe:**
      * Does the "Warm" heatmap light up the lamp?
      * Does the "Cool" heatmap light up the window?
      * **Success Metric:** If you see distinct (even if small) regions lighting up for different classes, your model is ready for Part 2. If the heatmaps are identical, the model failed to learn chromaticity features.

#### **Step 2: LSMI Data Preparation (Continuous to Discrete)**

The LSMI dataset provides **continuous** ground truth (pixel-wise mixture maps), but your model speaks a **discrete** language (5 clusters). [cite_start]You must bridge this gap using the "Discrete-to-Continuous Mapping Algorithm" outlined in the report[cite: 45].

**The Script: `prepare_lsmi_masks.py`**

**Inputs:**

  * LSMI Ground Truth (Mixture Maps $M$ and Illuminant Chromaticities $L$).
  * Part 1 Cluster Centroids ($\mu_1, \dots, \mu_5$) (from `export_centroids.py`).

**Algorithm:**

1.  **Load Centroids:** Load your 5 RGB centroids (Very Cool to Very Warm).
2.  **Iterate LSMI Images:** For each image with $N$ light sources:
      * **Get Illuminants:** Retrieve the ground truth RGB chromaticity for each light source $L_k$.
      * [cite_start]**Calculate Distances:** Compute the **Angular Error** between each light source $L_k$ and your 5 centroids ($\mu_j$)[cite: 49]:
        $$\Delta E_{ang} = \arccos \left( \frac{L_k \cdot \mu_j}{|L_k| |\mu_j|} \right) \times \frac{180}{\pi}$$
      * [cite_start]**Assign Labels:** Assign each light source $L_k$ to the *closest* cluster centroid[cite: 50].
          * *Example:* If Light Source A is $5^\circ$ from "Warm" and $20^\circ$ from "Neutral", Light Source A = "Warm".
      * [cite_start]**Filter Outliers:** If a light source is $>10^\circ$ away from *all* clusters, flag this image as "invalid" or "hard" (as suggested in the report)[cite: 51].
3.  **Generate Masks ($G_c$):**
      * Create 5 empty masks (one for each cluster).
      * [cite_start]For each pixel, sum the mixture ratios ($M$) of all light sources assigned to a specific cluster[cite: 53].
      * *Example:* If "Source 1" is assigned to "Warm", add Source 1's mixture map to the "Warm" mask.
4.  **Save:** Save these 5 channel masks (e.g., `.npz` format). [cite_start]These will serve as the "Ground Truth" for optimizing your Score-CAM and DenseCRF in the next steps[cite: 54, 96].