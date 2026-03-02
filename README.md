**Hybrid Intrusion Detection System for Encrypted Network Traffic**

This repository contains the implementation of a real-time, flow-based hybrid intrusion detection system (IDS) designed for encrypted network traffic environments (TLS). With the widespread adoption of encryption, traditional deep packet inspection is no longer viable, and detection must rely on payload-agnostic features extracted from network flows.

The system brings together two complementary approaches. A Random Forest classifier handles known attack patterns using supervised learning. Two Isolation Forest models, one for complete flows and one for partial flows, are trained on benign traffic to capture what normal behaviour looks like. These are combined in a confidence-gated architecture: when the supervised model makes a prediction with low confidence, that flow is passed to the appropriate Isolation Forest for a secondary assessment.

This design allows the system to maintain strong performance on known threats while remaining sensitive to novel or previously unseen behaviours. It also accounts for a practical reality of live traffic, connections are often incomplete or observed only for a short duration, by explicitly handling partial flows throughout the pipeline. The system is evaluated not only on offline test data but also under realistic streaming conditions, with attention to detection accuracy, false alarm rates, and operational feasibility.

**High-Level Design Architecture: Training and Real-Time Pathways**
<img width="965" height="635" alt="image" src="https://github.com/user-attachments/assets/5d2d3f82-59f4-4749-8e10-8beebd2f71b1" />

**Script Architecture & Processes**
<img width="858" height="1039" alt="image" src="https://github.com/user-attachments/assets/8746f2b1-0cd6-432a-b8ef-51a83ff493f5" />

**Repository Structure**
dissertation_ids_project/
│
├── data/
│   ├── raw/                    # Raw CSV flows from CICFlowMeter
│   ├── cleaned/                 # Preprocessed dataset (cleaned_dataset.csv)
│   └── processed/               # Train/val/test splits
│
├── models/
│   ├── rf_model.pkl             # Trained Random Forest
│   ├── iso_complete.pkl         # Isolation Forest (complete flows)
│   ├── iso_partial.pkl          # Isolation Forest (partial flows)
│   ├── scaler.pkl               # StandardScaler fitted on training data
│   ├── features.txt              # Ordered feature list
│   ├── label_mapping.json        # String-to-integer label mapping
│   └── thresholds.json           # Tuned thresholds (conf_gate, IF thresholds)
│
├── evaluation_results/           # All metrics, confusion matrices, plots
│   ├── metrics_*.json
│   ├── cm_*.png
│   ├── *_roc.png / *_pr.png
│   └── per_sample.csv
│
├── scripts/
│   ├── prepare_dataset.py        # Clean raw CSVs, remove identifiers
│   ├── split_trisplit.py         # Stratified 80/10/10 split
│   ├── train_hybrid_ids.py        # Train RF + dual IF models
│   ├── tune_thresholds.py         # Calibrate thresholds on validation set
│   ├── evaluate_baselines.py      # Final evaluation on test set
│   ├── generate_dataset_all.py    # Real-time flow merger (bi-directional + partial)
│   └── realtime_hybrid_ids.py     # Live inference on streaming flows
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE

Requirements
Python 3.12+
Ubuntu 22.04 LTS (recommended) or other Linux distribution
CICFlowMeter (for flow generation)
Dependencies listed in requirements.txt

**RESULT: Real-Time Hybrid IDS (Supervised & Unsupervised Learning Models)**
<img width="1063" height="426" alt="image" src="https://github.com/user-attachments/assets/0be5411e-f992-4c5b-8daa-09aa14b1e68d" />
