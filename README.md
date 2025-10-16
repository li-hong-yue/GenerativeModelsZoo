# 🧬 GenerativeModelsZoo

A collection of modern **generative models** including:
**VAE, VQ-VAE, GAN, WGAN, VQGAN, DDPM, DiT, CFG, Flow Matching, and VAR**.

---

## 🚀 Setup 

# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure paths & logging
#    - Update dataset and checkpoint paths in the config files.
#    - Log in to Weights & Biases:
wandb login

## 🧠 Training

# 3. Train a model
python training/train.py --config configs/{model}/{dataset}.yaml

# Example:
python training/train.py --config configs/ddpm/cifar10.yaml

## 📂 Structure

GenerativeModelsZoo/
├── configs/           # YAML configs for each model and dataset
├── models/            # Model definitions
├── training/          # Training scripts 
├── utils/             # Common utilities
├── evaluation/        # Evaluation scripts
└── requirements.txt   # Dependencies

## 📜 License

MIT License © 2025 GenerativeModelsZoo