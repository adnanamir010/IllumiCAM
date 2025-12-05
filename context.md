The objective of this research is to solve the **Multi-Illuminant Color Constancy** problem by repurposing a simple classification model into a complex spatial localizer.

Here is the detailed context on the objective for your LLM plan, broken down by the "Why", "What", and "How":

### 1. The High-Level "Why": Solving Real-World Lighting
**Current Limitation:** Traditional algorithms assume the entire world is lit by a single light source (e.g., just the sun). [cite_start]This fails in real photography and Augmented Reality (AR) where scenes often have mixed lighting (e.g., a room with a warm lamp *and* a cool window)[cite: 2, 3].
[cite_start]**The Goal:** To build a system that can look at an image and say, "The left side is lit by a tungsten bulb, but the right side is lit by daylight," so that color correction can be applied locally rather than globally[cite: 1, 6].

### 2. The Technical "What": Weakly Supervised Localization
[cite_start]The core technical objective is **Weakly Supervised Semantic Segmentation (WSSS)**[cite: 7].
* [cite_start]**The Input:** A Convolutional Neural Network (IlluminantCNN) that was trained *only* on simple labels like "This image is Warm" or "This image is Cool" (from Part 1)[cite: 8].
* **The Challenge:** The model was never told *where* the light is, nor was it trained on pixel-level data. [cite_start]Furthermore, its architecture (Max Pooling) actively discards spatial information to make classification easier[cite: 10, 19].
* [cite_start]**The Output:** You want to force this "blind" classifier to generate a high-resolution, pixel-perfect map showing exactly where each type of light is hitting the scene[cite: 6].

### 3. The Scientific Hypothesis
The project is testing a specific scientific hypothesis:
**"Discriminative classifiers implicitly learn segmentation."**
[cite_start]The hypothesis is that even though the model was only taught to shout "WARM!", its internal features actually learned to recognize the specific pixels of the "warm lamp" vs the "cool window" to make that decision[cite: 9]. The objective is to extract this hidden knowledge using:
1.  [cite_start]**Score-CAM:** To bypass the sparsity of Max Pooling and see the full region[cite: 25].
2.  [cite_start]**DenseCRF (Chromaticity):** To refine those regions by using the physics of light (chromaticity) to ignore shadows[cite: 68, 70].
3.  [cite_start]**LSMI Optimization:** To use ground-truth data not to train the model, but to tune the post-processing tools (the CRF)[cite: 73, 76].

**In summary for the LLM:**
"The objective is to take a pre-trained image classifier (IlluminantCNN) and unlock its latent spatial awareness to perform pixel-wise multi-illuminant estimation. This requires overcoming the information loss of Max Pooling using Score-CAM and refining the output with a physics-based DenseCRF, using the LSMI dataset as a benchmark for hyperparameter tuning."