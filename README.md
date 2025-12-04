```
Final_Project/
│
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── config.py                           # Configuration constants and hyperparameters
├── model.py                            # IlluminantCNN model definition
├── data_loader.py                      # Data loading utilities and transforms
│
├── train.py                            # Training script
├── evaluate.py                         # Evaluation script for test set
├── gradcam.py                          # Grad-CAM visualization script
├── illuminant_estimator.py            # Continuous illuminant estimation
│
├── augment_split_data.py               # Data augmentation and train/val/test splitting
├── visualize_image.py                  # Image visualization tool (CLI)
│
├── illuminant_eda.ipynb                # Exploratory Data Analysis notebook
├── training.ipynb                      # Original training notebook (reference)
│
├── cluster_centers.npy                 # Saved cluster centers (generated from EDA)
│
├── Data/                               # Raw dataset (excluded from git)
│   ├── Nikon_D810/
│   │   ├── field_1_cameras/           # Field images
│   │   ├── field_3_cameras/
│   │   ├── lab_printouts/
│   │   └── lab_realscene/
│   │       ├── *.tiff                  # Image files
│   │       └── *.wp                    # White point files
│   └── info/                           # Camera characterization data
│       ├── Info/
│       │   ├── reference_wps_ccms_nikond810.mat
│       │   └── ...
│       └── Nikon_D810_Info/
│           └── ...
│
├── dataset/                            # Generated dataset (excluded from git)
│   ├── train/
│   │   ├── Cool/
│   │   ├── Neutral/
│   │   ├── Very_Cool/
│   │   ├── Very_Warm/
│   │   └── Warm/
│   ├── val/
│   │   └── [same structure as train]
│   └── test/
│       └── [same structure as train]
│
└── visualizations/                     # Generated visualizations (excluded from git)
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── gradcam_grid_all_classes.png
    ├── illuminant_estimation_results.png
    ├── illuminant_examples.png
    └── ...
```