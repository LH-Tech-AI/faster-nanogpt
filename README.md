# 🚀 Faster nanoGPT

This repository is a modern, high-performance evolution of Karpathy's original **nanoGPT**. It integrates state-of-the-art architecture improvements and the revolutionary **Muon optimizer** to achieve faster convergence and better loss plateaus.

---

## 🌟 Key Improvements

* **Muon Optimizer:** Uses Newton-Schulz orthogonalization for 2D weights, leading to significantly higher sample efficiency.
* **Modern Architecture:**
    * **RoPE** (Rotary Positional Embeddings) for better context handling.
    * **RMSNorm & QK-Norm** for enhanced training stability and speed.
    * **ReLU² Activation** for improved non-linearity in small-to-medium models.
    * **Logit Soft-Capping** (Gemma-2 style) to prevent training instability.
* **Blackwell-Ready:** Optimized for `bfloat16` and `torch.compile` on modern GPUs (RTX 40/50 series).

---

## 📊 Training Performance (nanoGPT vs. Optimized)

In our benchmarks (**7.23M parameter model** on TinyStories), the optimized version achieved the same validation loss as the original in nearly half the time.

| Metric | Original nanoGPT (AdamW) | Optimized nanoGPT (Muon) | Improvement |
| :--- | :--- | :--- | :--- |
| **Convergence (Loss < 2.0)** | 3,140 Iters | 2,090 Iters | ~33% fewer steps |
| **Efficiency** | 1.0x | ~1.6x | More knowledge per token |

This benchmark shows that faster-nanogpt is approximately **1.6x faster** at dropping the loss, which translates to a **60% improvement** in training efficiency compared to the original implementation of nanoGPT.

![Graph that shows the training of the above model with orginal nanoGPT and faster-nanogpt](https://raw.githubusercontent.com/LH-Tech-AI/faster-nanogpt/refs/heads/main/images/graph-loss.png)

---

## 🛠️ Quick Start (Home Setup)
First, you'll have to install these dependencies for Python:
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install tiktoken tqdm datasets numpy pillow
```
If you are on Linux, I recommend using a Python venv:
```bash
python -m venv venv
```
Then, you have to clone the faster-nanogpt repo from GitHub - for example using git:
```bash
git clone https://github.com/LH-Tech-AI/faster-nanogpt.git
cd faster-nanogpt
```
Once you have cloned the repo and have all setup, you can start the real training.
This an example for training a 7.23M parameter model on TinyStories with torch.compile on your GPU:
1. Prepare the data:
```bash
python3 prepare.py
```
2. Start the training:
```bash
python3 train.py   --dataset=tinystories   --n_layer=4   --n_head=4   --n_embd=128   --block_size=256   --batch_size=32   --gradient_accumulation_steps=4   --learning_rate=1e-3   --max_iters=5000   --eval_interval=250   --eval_iters=50   --log_interval=10   --weight_decay=0.1   --warmup_iters=200   --lr_decay_iters=5000   --min_lr=1e-4   --dtype=float32   --compile=True   --always_save_checkpoint=True   --init_from=scratch   --out_dir=out/tinystories_7m   --use_muon=True   --muon_lr=0.02   --muon_momentum=0.95   --schedule=trapezoidal   --cooldown_iters=1000   --use_rope=True   --activation=relu2   --norm_type=rmsnorm   --qk_norm=True   --logit_soft_cap=30.0   --dropout=0.0   --bias=False   --grad_clip=1.0
```
3. Watch the training:
- Did it compile correctly without errors?
- Did it run to the 10th iteration without CUDA OOM error?
- Does it run in less than 1 or 2 seconds per iteration?
Then everything is fine! You've done it.

If you are running this on a modern GPU (like an RTX 5060 Ti or 4090), make sure to use:

```bash
python train.py --dtype=bfloat16 --compile=True --use_muon=True
```
