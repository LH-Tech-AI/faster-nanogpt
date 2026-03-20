"""
Optimized training script for nanoGPT with modern improvements.

Key improvements over original nanoGPT:
- Muon optimizer for hidden layers (much faster convergence)
- Trapezoidal LR schedule (easier to tune, matches cosine performance)
- Modern architecture: RoPE, RMSNorm, QK-Norm, ReLU², logit soft-capping
- All original functionality preserved (DDP, wandb, checkpoints, etc.)

To run on a single GPU:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

torch.set_float32_matmul_precision('high')

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model architecture
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # no bias in Linear/Norm layers (modern default)
use_rope = True # RoPE instead of learned position embeddings
rope_base = 10000.0
activation = 'relu2' # 'relu2' or 'gelu'
norm_type = 'rmsnorm' # 'rmsnorm' or 'layernorm'
qk_norm = True # QK-Norm for attention stability
logit_soft_cap = 30.0 # soft-capping for logits, 0.0 to disable
# optimizer
use_muon = True # use Muon optimizer for hidden layer weights
muon_lr = 0.02 # Muon learning rate for hidden 2D params
muon_momentum = 0.95 # Muon momentum factor
learning_rate = 6e-4 # AdamW learning rate for embedding/head/scalar params
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
schedule = 'trapezoidal' # 'trapezoidal' or 'cosine'
warmup_iters = 2000 # how many steps to warm up for
cooldown_iters = 0 # for trapezoidal: 0 = auto (20% of max_iters)
lr_decay_iters = 600000 # for cosine: should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    # Check if directory exists and is writable
    if not os.path.isdir(out_dir):
        raise RuntimeError(f"ERROR: out_dir {out_dir} could not be created!")
    
    test_file = os.path.join(out_dir, '.write_test')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"out_dir {out_dir} is ready and writable.")
    except (IOError, OSError) as e:
        raise RuntimeError(f"ERROR: out_dir {out_dir} not writable! Details: {e}")
torch.manual_seed(1337 + seed_offset)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # 1. Wir stacken erst die nativen uint16 (weniger CPU Last)
    x_raw = torch.stack([torch.from_numpy(data[i:i+block_size]) for i in ix])
    y_raw = torch.stack([torch.from_numpy(data[i+1:i+1+block_size]) for i in ix])
    
    if device_type == 'cuda':
        # 2. Wir schicken uint16 zur GPU (spart Bandbreite) 
        # und konvertieren ERST DORT zu long (int64)
        x = x_raw.to(device, dtype=torch.long, non_blocking=True)
        y = y_raw.to(device, dtype=torch.long, non_blocking=True)
    else:
        x = x_raw.to(device, dtype=torch.long)
        y = y_raw.to(device, dtype=torch.long)
        
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout,
    use_rope=use_rope, rope_base=rope_base, activation=activation,
    norm_type=norm_type, qk_norm=qk_norm, logit_soft_cap=logit_soft_cap,
)

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # Force structural config from checkpoint
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # Load architecture config from checkpoint (with backward-compat defaults for old checkpoints)
    legacy_defaults = {
        'use_rope': False, 'activation': 'gelu', 'norm_type': 'layernorm',
        'qk_norm': False, 'logit_soft_cap': 0.0, 'rope_base': 10000.0,
    }
    for k in ['use_rope', 'activation', 'norm_type', 'qk_norm', 'logit_soft_cap', 'rope_base']:
        model_args[k] = checkpoint_model_args.get(k, legacy_defaults[k])
    # Create model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # Fix keys from torch.compile checkpoints
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # GPT-2 requires legacy architecture
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # Read created config params for checkpointing
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
    # Also store legacy architecture settings
    for k in ['use_rope', 'activation', 'norm_type', 'qk_norm', 'logit_soft_cap', 'rope_base']:
        model_args[k] = getattr(model.config, k)

# crop down the model block size if desired
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type,
    use_muon=use_muon, muon_lr=muon_lr, muon_momentum=muon_momentum,
)
# Store base_lr for scheduling (in case configure_optimizers didn't set them)
for g in optimizer.param_groups:
    if 'base_lr' not in g:
        g['base_lr'] = g['lr']

if init_from == 'resume':
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Restore base_lr after loading state dict
        for g in optimizer.param_groups:
            if 'base_lr' not in g:
                g['base_lr'] = g['lr']
    except (ValueError, KeyError, RuntimeError) as e:
        print(f"Warning: could not load optimizer state: {e}")
        print("Starting optimizer from scratch (model weights were loaded)")
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler
def get_lr_multiplier(it):
    """
    Compute learning rate multiplier (0 to 1) for the current iteration.
    Applied to each param group's base_lr.
    """
    min_lr_ratio = min_lr / learning_rate if learning_rate > 0 else 0.0

    if schedule == 'trapezoidal':
        # Warmup phase: linear ramp up
        if it < warmup_iters:
            return (it + 1) / (warmup_iters + 1)
        # Cooldown phase: linear ramp down
        actual_cooldown = cooldown_iters if cooldown_iters > 0 else int(0.2 * max_iters)
        cooldown_start = max_iters - actual_cooldown
        if it >= cooldown_start:
            progress = (it - cooldown_start) / max(1, actual_cooldown)
            return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
        # Constant phase
        return 1.0
    else:
        # Cosine schedule (original behavior)
        if it < warmup_iters:
            return (it + 1) / (warmup_iters + 1)
        if it > lr_decay_iters:
            return min_lr_ratio
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr_ratio + coeff * (1.0 - min_lr_ratio)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

while True:

    # determine and set the learning rate for this iteration
    if decay_lr:
        mult = get_lr_multiplier(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['base_lr'] * mult
        lr = learning_rate * mult  # for logging (AdamW base rate * multiplier)
    else:
        lr = learning_rate

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,
            }
            if use_muon:
                muon_lr_current = optimizer.param_groups[0]['lr']
                log_dict["muon_lr"] = muon_lr_current
            wandb.log(log_dict)

        if losses['val'] < best_val_loss or always_save_checkpoint:
            # WICHTIG: Wir aktualisieren best_val_loss noch NICHT hier oben, 
            # damit die if-Abfrage unten den Vergleich noch machen kann.
            
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss, # der bisherige Bestwert
                    'config': config,
                }
    
                # 1. Speichere IMMER den aktuellsten Stand
                print(f"saving latest checkpoint to {os.path.join(out_dir, 'ckpt.pt')}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
                # 2. Speichere den BESTEN Stand nur, wenn er wirklich besser ist
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val'] # Jetzt erst aktualisieren
                    print(f"new best val loss: {best_val_loss:.4f}! saving best_ckpt.pt")
                    torch.save(checkpoint, os.path.join(out_dir, 'best_ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
