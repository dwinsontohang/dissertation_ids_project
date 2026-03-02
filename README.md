**Hybrid Intrusion Detection System for Encrypted Network Traffic**

This repository contains the implementation of a real-time, flow-based hybrid intrusion detection system (IDS) designed for encrypted network traffic environments (TLS). With the widespread adoption of encryption, traditional deep packet inspection is no longer viable, and detection must rely on payload-agnostic features extracted from network flows.

The system brings together two complementary approaches. A Random Forest classifier handles known attack patterns using supervised learning. Two Isolation Forest models, one for complete flows and one for partial flows, are trained on benign traffic to capture what normal behaviour looks like. These are combined in a confidence-gated architecture: when the supervised model makes a prediction with low confidence, that flow is passed to the appropriate Isolation Forest for a secondary assessment.

This design allows the system to maintain strong performance on known threats while remaining sensitive to novel or previously unseen behaviours. It also accounts for a practical reality of live traffic, connections are often incomplete or observed only for a short duration, by explicitly handling partial flows throughout the pipeline. The system is evaluated not only on offline test data but also under realistic streaming conditions, with attention to detection accuracy, false alarm rates, and operational feasibility.


**High-Level Design Architecture: Training and Real-Time Pathways**

<img width="965" height="635" alt="image" src="https://github.com/user-attachments/assets/5d2d3f82-59f4-4749-8e10-8beebd2f71b1" />


**Script Architecture & Processes**

<img width="858" height="1039" alt="image" src="https://github.com/user-attachments/assets/8746f2b1-0cd6-432a-b8ef-51a83ff493f5" />


**Requirements**

- Python 3.12+
- Ubuntu 22.04 LTS (recommended) or other Linux distribution
- CICFlowMeter (for flow generation)
- Dependencies listed in requirements.txt


**Sceenshoot Result: Real-Time Hybrid IDS (Supervised & Unsupervised Learning Models)**

<img width="1063" height="426" alt="image" src="https://github.com/user-attachments/assets/0be5411e-f992-4c5b-8daa-09aa14b1e68d" />

The output displays each incoming flow as it is processed, showing the final classification decision, confidence scores from the Random Forest, anomaly scores from the Isolation Forest, and whether the flow was handled directly by the supervised model or routed to the unsupervised layer. When an attacker performs a brute-force attack (right side), the IDS detects it in real-time (left side), as shown by the classifications appearing immediately as the attack traffic flows in.
