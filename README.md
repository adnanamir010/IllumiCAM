```
Final_Project/
│
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python dependencies
│
├── augment_split_data.py               # Data augmentation and train/val/test splitting
├── visualize_image.py                  # Image visualization tool (CLI)
│
├── illuminant_eda.ipynb                # Exploratory Data Analysis notebook
├── training.ipynb                      # Model training and evaluation notebook
│
├── cluster_centers.npy                 # Saved cluster centers (generated from EDA)
│
├── Data/                               # Raw dataset (excluded from git)
│   ├── Nikon_D810/
│   │   ├── field_1_cameras/           # Field images
│   │   ├── field_3_cameras/
│   │   ├── lab_printouts/
│   │   └── lab_realscene/
│   │       ├── *.tiff                # Image files
│   │       └── *.wp                   # White point files
│   └── info/                          # Camera characterization data
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
├── visualizations/                     # Generated visualizations (excluded from git)
│   ├── gradcam_grid_all_classes.png
│   ├── confusion_matrix.png
│   └── ...
│
├── gradcam_heatmaps/                   # GradCAM outputs (excluded from git)
│   └── ...
│
└── new_venv/                          # Virtual environment (excluded from git)
    └── ...
```
