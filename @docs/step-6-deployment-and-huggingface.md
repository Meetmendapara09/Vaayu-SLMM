# Step 6: Deployment, Hugging Face Hub Release & GitHub Synchronization

## 1. Overview

To conclude the lifecycle of **Vaayu SLMM**, the model is published across two complementary platforms:
1. **GitHub Private Repository (`Meetmendapara09/Vaayu-SLMM`)**:
   - Houses the complete source code, tokenizer definitions, synthetic MCP data generators, embedded engine, and documentation.
2. **Hugging Face Hub (Public Repository)**:
   - Live repository: 👉 [**`https://huggingface.co/meetmendapara/Vaayu-Base`**](https://huggingface.co/meetmendapara/Vaayu-Base)
   - Hosts the base model weights, BPE vocabulary, configuration, and public Model Card.

---

## 2. Pushing to Hugging Face Hub

### Step 2.1: Authentication
Login to Hugging Face using the CLI or retrieve an access token with `write` permission from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):

```bash
python -m huggingface_hub.cli.hf auth login
```

### Step 2.2: Uploading Model & Weights
Run the automated packaging and publishing script:

```bash
python scripts/push_to_hf.py --repo_id <YOUR_HF_USERNAME>/Vaayu --checkpoint checkpoints/vaayu_final.pt
```

Or pass your token directly:
```bash
python scripts/push_to_hf.py --repo_id <YOUR_HF_USERNAME>/Vaayu --token <HF_WRITE_TOKEN>
```

The script automatically:
- Creates the repository on Hugging Face Hub as a **public** repository.
- Bundles `README.md` (complete Model Card with YAML tags), `config.json`, and all tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`).
- Uploads the trained model weights `vaayu_final.pt`.

---

## 3. GitHub Private Repository Maintenance

The private GitHub repository `Meetmendapara09/Vaayu-SLMM` is updated continuously through every development step:

```bash
git add .
git commit -m "Update Vaayu model and documentation"
git push origin main
```