# TruthLens Video Forensics — Pretrained Model Directory

This directory contains the lightweight neural network weights and configuration files for TruthLens:

- `ai_gen_efficientnet.onnx` (640 KB) — AI generation detector
- `face_detection_yunet_2023mar.onnx` (220 KB) — YuNet face detector
- `meso4_df.pt` (120 KB) — MesoNet DeepFake classifier
- `meso4_f2f.pt` (120 KB) — MesoNet Face2Face classifier
- `cifake/model.onnx` (640 KB) — CIFAKE EfficientNet-B0 ONNX structure
- `cifake/model.onnx.data` (15.25 MB) — CIFAKE ONNX tensor weights

## Heavy Model Weights (Exceeding GitHub 100 MB Limit)
The following binary weights exceed GitHub's 100 MB file size limit and are downloaded on demand:
- `gend_clip_vit_l14.safetensors` (1.16 GB) — HuggingFace: `yermandy/GenD_CLIP_L_14`
- `xception-b5690688.pth` (87.4 MB) — PyTorch Hub XceptionNet

### Downloading Heavy Checkpoints:
Run:
```bash
python scripts/download_models.py
```
Or place the downloaded weights directly into this folder.
The platform automatically operates with graceful degradation (traditional visual forensics, MesoNet, YuNet, and audio deepfake detection remain 100% active) even if these heavy weights are not locally present.
