# 🚀 ImageNet-1K Training on AWS EC2: A Cloud Computing Journey

## 🎯 What's This All About?

Ever wondered how those fancy AI models get trained? Well, buckle up! This project is all about training a **ResNet50** deep learning model from scratch on the massive **ImageNet-1K dataset** (that's 1.2 million images across 1000 categories!) using **AWS EC2 cloud infrastructure**. 

Think of it like teaching a computer to recognize everything from golden retrievers to sports cars, and doing it all in the cloud! ☁️

---

## 🏆 The Results (Drumroll, Please!)

After **4 days and a few hours** of non-stop GPU crunching on AWS EC2, here's what we achieved:

| Metric | Score | What It Means |
|--------|-------|---------------|
| **Top-1 Accuracy** | **76.77%** | The model's first guess is correct 76.77% of the time! 🎯 |
| **Top-5 Accuracy** | **93.51%** | The correct answer is in the top 5 guesses 93.51% of the time! 🔥 |
| **Training Epochs** | 90 | We ran through the entire dataset 90 times |
| **Total Training Time** | ~4 days | That's 96+ hours of pure AI learning! |

These numbers are pretty solid for training from scratch without using any pre-trained weights!

---

## 🛠️ AWS Services Used

### 1️⃣ **Amazon EC2 (Elastic Compute Cloud)** - The Powerhouse 💪
- **Instance Type**: g6xlarge (GPU-powered beast)
- **What it does**: Provides the heavy-duty GPU muscle needed to train our model
- **Why we chose it**: Training deep learning models needs serious computational power. This GPU instance made it possible to train on 1.2M images without waiting months!
- **Cost Efficiency**: We used `screen` sessions so we could disconnect our laptop and let training run uninterrupted (no babysitting needed!)

### 2️⃣ **Amazon S3 (Simple Storage Service)** - The Data Vault 📦
- **What we stored**: 
  - Training checkpoints (model snapshots)
  - Final model weights
  - Dataset backups
  - Training logs and metrics
- **Why it's awesome**: Cheap, reliable storage that doesn't disappear when you terminate your EC2 instance
- **Usage**: Automatically sync checkpoints to S3 during training for backup and easy sharing

### 3️⃣ **AWS Lambda + S3** - The Serverless Showcase 🌐
- **What it does**: Hosts our interactive demo website (HTML/CSS/JS) that explains the project
- **Why it's cool**: 
  - Pay ONLY when someone visits (pennies per request!)
  - No server management needed
  - Integrates perfectly with S3 for static hosting
  - **Free tier eligible** - First million requests per month are FREE!
- **What visitors see**: Beautiful, user-friendly interface explaining our journey, metrics, and model architecture

---

## 📁 Project Structure

```
imagenet-1k/
├── mini-project/
│   ├── configs/                      # Configuration files for different setups
│   │   ├── imagenet1k_config.yaml   # Main training config
│   │   ├── g6_config.yaml           # EC2-specific config
│   │   └── colab_config.yaml        # Google Colab config (backup)
│   │
│   ├── src/                          # Source code magic ✨
│   │   ├── models/
│   │   │   └── resnet.py            # ResNet50 architecture
│   │   ├── training/
│   │   │   └── trainer.py           # Training loop logic
│   │   └── utils/
│   │       ├── logger.py            # WandB integration
│   │       ├── metrics.py           # Accuracy calculators
│   │       ├── losses.py            # Loss functions
│   │       └── optimizers.py        # SGD + Cosine scheduler
│   │
│   ├── data-download-scripts/        # Dataset preparation scripts
│   │   └── imagenet-1k/
│   │       ├── hf_donload.py        # Download from HuggingFace
│   │       ├── arrange_val.py       # Organize validation set
│   │       └── arrange.py           # Organize training set
│   │
│   ├── train.py                      # Main training script
│   ├── evaluate.py                   # Model evaluation
│   ├── requirements.txt              # Python dependencies
│   ├── training_best.log             # Full training metrics
│   └── resnet50_imagenet1k_cpu.pth  # Final trained weights 🏆
│
└── huggingface-space/                # Interactive demo
    ├── app.py                        # Gradio web interface
    └── requirements.txt              # Demo dependencies
```

---

## 🎬 The Training Pipeline (Step-by-Step)

### Step 1: Dataset Download 📥
```bash
python data-download-scripts/imagenet-1k/hf_donload.py
```
Downloaded the entire ImageNet-1K dataset from HuggingFace (that's ~140GB of images!)

### Step 2: Data Organization 📂
```bash
python data-download-scripts/imagenet-1k/arrange_val.py
python data-download-scripts/imagenet-1k/arrange.py
```
Organized 1.2M training images and 50K validation images into proper folder structures. Each of the 1000 classes got its own folder!

### Step 3: Launch Training 🚂
```bash
screen -S training  # Create a persistent session
python train.py --config configs/imagenet1k_config.yaml
# Press Ctrl+A then D to detach
```
Started training and detached the screen session so we could:
- Close our laptop ✅
- Sleep peacefully 😴
- Check progress anytime via WandB dashboard 📊

### Step 4: Monitor with WandB 📈
Connected to WandB AI (Weights & Biases) for:
- Real-time loss and accuracy graphs
- System metrics (GPU usage, memory)
- Training speed stats
- No need to SSH into EC2 constantly!

### Step 5: Save & Deploy 🎉
- Best model weights saved automatically
- Uploaded to HuggingFace for easy sharing
- Deployed demo on AWS Lambda

---

## 🧪 Training Configuration

```yaml
Model: ResNet50 (25.6M parameters)
Dataset: ImageNet-1K (1.28M training images, 50K validation)
Batch Size: 128
Optimizer: SGD with Nesterov momentum (0.9)
Learning Rate: 0.1 → 1e-5 (cosine decay with 5 epochs warmup)
Regularization: 
  - Weight Decay: 1e-4
  - Label Smoothing: 0.1
  - Mixup (α=0.2)
  - CutMix (α=1.0)
  - Drop Path: 0.1
Training: 90 epochs (~4 days on EC2 g6xlarge)
Mixed Precision: Enabled (faster training!)
```

---

## 🎨 Key Features & Techniques

### 1. **Data Augmentation** 🖼️
- Random resizing and cropping
- Horizontal flipping
- Mixup: Blends two images together
- CutMix: Pastes patches from one image onto another
- **Why?** Makes the model robust to variations

### 2. **Advanced Training Tricks** 🎓
- **Label Smoothing**: Prevents overconfidence
- **Cosine Annealing**: Gradually reduces learning rate
- **Warmup**: Starts with tiny learning rate for stability
- **Drop Path**: Regularization technique for better generalization
- **Mixed Precision (AMP)**: Trains faster using FP16

### 3. **Cloud-Native Design** ☁️
- Automatic checkpoint saving
- Resume training from any epoch
- WandB integration for remote monitoring
- Screen sessions for disconnect-proof training

---

## 📊 Training Progress Highlights

| Epoch | Train Top-1 | Val Top-1 | Train Top-5 | Val Top-5 |
|-------|-------------|-----------|-------------|-----------|
| 0 | 2.77% | 9.16% | 8.65% | 24.15% |
| 30 | 56.33% | 60.68% | 79.41% | 83.91% |
| 60 | 64.39% | 68.86% | 84.94% | 89.14% |
| **88** | **75.33%** | **76.77%** | **91.09%** | **93.51%** |

Watch how the model goes from basically guessing to actually understanding images! 🤯

---

## 🌐 Live Demo & Resources

### 🤗 HuggingFace Space
- **Interactive Demo**: Upload any image and get instant predictions!
- **Model Weights**: Download the trained model
- **Link**: [Coming Soon - Your HuggingFace Space Link]

### 🎨 AWS Lambda Static Site
- **Beautiful UI**: Clean, modern interface explaining everything
- **Hosted on**: AWS Lambda + S3 (serverless = FREE!)
- **Features**:
  - Project overview
  - Training metrics visualization
  - Architecture diagrams
  - Step-by-step guide
- **Link**: [Coming Soon - Your Lambda URL]

---

## 💰 Cost Breakdown (Approximate)

| Service | Usage | Estimated Cost |
|---------|-------|----------------|
| EC2 g6xlarge | ~100 hours | $150-200 |
| S3 Storage | ~200GB | $5/month |
| Data Transfer | ~150GB | $10-15 |
| Lambda | Static hosting | **FREE** (under 1M requests) |
| **Total** | Full project | **~$165-220** |

Pro tip: Stop your EC2 instance when not training to save $$!

---

## 🚀 How to Reproduce

### Prerequisites
- AWS Account with EC2 access
- HuggingFace account (for dataset)
- WandB account (for monitoring)

### Quick Start
```bash
# 1. Clone the repo
git clone <your-repo-url>
cd imagenet-1k/mini-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up WandB
echo "WANDB_API_KEY=your_key_here" > .env

# 4. Download dataset
python data-download-scripts/imagenet-1k/hf_donload.py
python data-download-scripts/imagenet-1k/arrange_val.py
python data-download-scripts/imagenet-1k/arrange.py

# 5. Start training
screen -S training
python train.py --config configs/imagenet1k_config.yaml
# Ctrl+A then D to detach

# 6. Monitor on WandB dashboard
# Visit: https://wandb.ai/your-project
```

---

## 🎓 What I Learned

1. **Cloud Computing is Powerful**: EC2 GPUs are insanely fast compared to local machines
2. **Patience is Key**: 4 days of training teaches you patience! 😅
3. **Monitoring Matters**: WandB made remote training actually manageable
4. **Screen Sessions Save Lives**: No more "oops, I closed the terminal" moments
5. **AWS is Versatile**: From compute (EC2) to storage (S3) to serverless (Lambda)
6. **Data Prep is 50% of the Work**: Organizing 1.2M images is no joke!

---

## 🙏 Acknowledgments

- **ImageNet**: For the amazing dataset
- **HuggingFace**: For easy dataset access
- **PyTorch**: For the awesome deep learning framework
- **WandB**: For incredible experiment tracking
- **AWS**: For the cloud infrastructure
- **Coffee**: For keeping me awake during late-night monitoring ☕

---

## 📝 Future Improvements

- [ ] Try larger models (ResNet101, EfficientNet)
- [ ] Experiment with different augmentation strategies
- [ ] Deploy full inference API on AWS API Gateway
- [ ] Add explainability (Grad-CAM visualizations)
- [ ] Compare with transfer learning approaches
- [ ] Try distributed training across multiple GPUs

---

## 📧 Contact & Contributions

Have questions? Want to collaborate? Feel free to reach out!

**Made with 💙 and lots of GPU power!**

---

## 📜 License

This project is open source and available under the MIT License.

---

### ⭐ If this project helped you learn about cloud computing or deep learning, give it a star!


