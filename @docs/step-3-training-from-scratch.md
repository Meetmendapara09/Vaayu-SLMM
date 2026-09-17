# Step 3: Training from Scratch on Kaggle Online GPUs

## 1. Overview

Vaayu is a **base real model** trained from scratch—it is **not** an existing open-source checkpoint (like LLaMA, Mistral, or Qwen) with fine-tuning or LoRA applied. 

To honor the **under 12-hour compute limit** while training a **245M parameter** model:
- The training job executes remotely on **Kaggle accelerated online GPUs** (NVIDIA T4 / P100 with high-bandwidth memory).
- Automatic Mixed Precision (`torch.amp.autocast`) with FP16 gradients reduces compute time by ~2.5x.
- Sequence packing and micro-batching with gradient accumulation maximize GPU tensor core utilization.

---

## 2. Kaggle CLI Execution & Pipeline

The Kaggle training package is located in `kaggle_run/`:
- `kaggle_run/kernel-metadata.json`: Declares GPU accelerator, dependencies, and datasets.
- `kaggle_run/train_on_kaggle.py`: Complete self-contained scratch training loop.

### Pushing Training Job via Kaggle CLI:
```bash
kaggle kernels push -p kaggle_run
```

### Checking Status:
```bash
kaggle kernels status mendaparameet/vaayu-slmm-training
```
When running, Kaggle reports:
```
mendaparameet/vaayu-slmm-training has status "KernelWorkerStatus.RUNNING"
```

### Retrieving Trained Weights:
Once the status changes to `"KernelWorkerStatus.COMPLETE"`:
```bash
kaggle kernels output mendaparameet/vaayu-slmm-training -p checkpoints/
```

---

## 3. Local Training Engine Fallback

For local experimentation or micro-iteration, `src/training/train.py` provides a full training loop:
```bash
python src/training/train.py --epochs 4 --batch_size 4 --gradient_accumulation_steps 8 --max_hours 11.5
```