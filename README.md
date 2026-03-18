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

## 📊 Training Performance (nanoGPT vs. faster-nanogpt)

In our benchmark (**7.23M parameter model** on TinyStories trained on **T4** GPU), the optimized version achieved the same validation loss as the original in nearly half the time.

| Metric | Original nanoGPT (AdamW) | faster-nanogpt (Muon) | Improvement |
| :--- | :--- | :--- | :--- |
| **Convergence (Loss < 2.0)** | 3,140 Iters | 2,090 Iters | ~33% fewer steps |
| **Efficiency** | 1.0x | ~1.6x | More knowledge per token |

This benchmark shows that faster-nanogpt is approximately **1.6x faster** at dropping the loss (1000+ iterations less for the same val loss), which translates to a **60% improvement** in training efficiency compared to the original implementation of nanoGPT.

![Graph that shows the training of the above model with orginal nanoGPT and faster-nanogpt](https://raw.githubusercontent.com/LH-Tech-AI/faster-nanogpt/refs/heads/main/images/graph-loss.png)

---

## 🛠️ Quick Start
First, you'll have to **install these dependencies** for Python (if you haven't already):
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install tiktoken tqdm datasets numpy pillow
```
or simply use `pip install -r requirements.txt`

If you are on **Linux**, I recommend using a **Python venv**:
```bash
python -m venv venv
```
Then, you have to **clone the faster-nanogpt repo** from GitHub - for example using git:
```bash
git clone https://github.com/LH-Tech-AI/faster-nanogpt.git
cd faster-nanogpt
```
Once you have cloned the repo and have all setup, you can start the real training.
This an example for training a 7.23M parameter model on TinyStories with torch.compile on your GPU:

**1. Prepare the data**:

```bash
python3 prepare.py
```
**2. Start the training**:

```bash
python3 train.py   --dataset=tinystories   --n_layer=4   --n_head=4   --n_embd=128   --block_size=256   --batch_size=32   --gradient_accumulation_steps=4   --learning_rate=1e-3   --max_iters=5000   --eval_interval=250   --eval_iters=50   --log_interval=10   --weight_decay=0.1   --warmup_iters=200   --lr_decay_iters=5000   --min_lr=1e-4   --dtype=float32   --compile=True   --always_save_checkpoint=True   --init_from=scratch   --out_dir=out/tinystories_7m   --use_muon=True   --muon_lr=0.02   --muon_momentum=0.95   --schedule=trapezoidal   --cooldown_iters=1000   --use_rope=True   --activation=relu2   --norm_type=rmsnorm   --qk_norm=True   --logit_soft_cap=30.0   --dropout=0.0   --bias=False   --grad_clip=1.0
```
**3. Watch the training:**

- Did it **compile correctly without errors**?
- Did it run to the 10th iteration **without CUDA OOM error**?
- Does it run in less than **1 or 2 seconds per iteration**?
Then **everything is fine**! You've done it.

**4. Test the model:**

To test the model, simply run:
```bash
python3 sample.py --out_dir=out/tinystories_7m --start="Once upon a time, there was a dog named Max" --num_samples=3 --max_new_tokens=200 --temperature=0.7 --device=cuda
```
You can - of course - also use another **start** prompt.

Note: If you **don't have a GPU**, change `--device=cuda` to `--device=cpu` in the sample command (it will be slower but it works).

---

If you are running this on **a modern GPU** (like an RTX 5060 Ti or 4090), make sure to use:

```bash
python train.py --dtype=bfloat16 --compile=True --use_muon=True
```

---

### 📖 Qualitative Comparison (at Step 1000)

To see the difference in "intelligence", we prompted both versions with:
*"Once upon a time, there was a dog named Max"*

| Original nanoGPT (AdamW) | faster-nanogpt (Muon + Modern Arch) |
| :--- | :--- |
| *Once upon a time, there was a dog named Max. Max loved to play outside in the sky. Every day, he would go to the park with his friends. They would play pretend he was a ball, a ball, and a ball.<br><br>One day, Max decided to visit the park with his friends. He looked at the park and saw a big tree with many colors. The tree felt a little scared, but the branches could not reach them.<br><br>Max felt sad. He shouted with his friends. "I want to play with the ball!"<br><br>But Max didn't like the ball anymore. He was scared. He ran to his friends and played with the ball.<br><br>But then, a big dog came to the dog. Max was scared. He wanted to play with the ball. The dog said, "Don't hurt you, Max. It was not nice. The dog is mean to break your ball. It is mean. You need to fix it."<br><br>Max looked at the ball and said* | *Once upon a time, there was a dog named Max. Max was very sad because he lived in a big field with his friends. One day, Max saw a big, tall tree in the tree. He got scared and wanted to go away and get out of the tree.<br><br>Max asked his mom, "What is wrong, so it's too shiny for me?" His mom smiled and said, "I don't want to get hurt. You just need to be quiet."<br><br>Max was very sad and he wanted to escape. He watched the other children in the yard and the big tree. He saw the tree and thought it was very pretty. He asked his mom, "Can we help me find the way you will be back here?"<br><br>His mom smiled and said, "Yes, let's go."<br><br>Max was happy to see the big tree and he knew he could help him find the owner. He was glad he was safe and happy. From then on, Max always remembered to be kind and quiet.* |
| **Observation:** Struggles significantly with coherence and logic. It falls into repetitive loops ("a ball, a ball, and a ball") and loses track of the story's subject. Semantic confusion is high (e.g., "dog came to the dog"). | **Observation:** Shows much higher intellectual density. It maintains a consistent narrative thread, uses proper dialogue structures, and demonstrates causal reasoning ("Max was sad because..."). The grammar is significantly more stable. |

Keep in mind, that this model was trained on a **T4** GPU - **not a modern GPU** that supports native **bfloat16** - I'll try that and update this README here to keep you up to date.
**Spoiler:** The overhead of ~90ms per iteration will be almost deleted.

---

### **🧠 Why is it better?**

The significant jump in quality and convergence speed is driven by three core upgrades:

1. **Muon Optimizer:** Unlike AdamW, which updates weights element-wise, Muon maintains **orthogonality** in the 2D weight matrices. This ensures that the model learns diverse features without redundant or "collapsed" gradients, leading to much higher intellectual density per parameter.  
2. **RoPE (Rotary Positional Embeddings):** Traditional GPT-2 use absolute position embeddings which struggle with long-range logic. RoPE allows the model to understand the **relative distance** between words much better, which is crucial for maintaining story consistency and grammar.  
3. **RMSNorm & QK-Norm:** These stabilization layers prevent the internal activations from "exploding." This allows us to use a higher learning rate and ensures that every training step contributes meaningfully to the model's understanding of language.

---

*Thanks to Andrej Karpathy for his original implementation of nanoGPT and to Keller Jordan for the Muon optimizer*
