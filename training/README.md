# Fine-tuning a teacher's voice (VoxCPM2 LoRA, on a Mac)

These steps take a teacher's long recordings and produce a VoxCPM2 model fine-tuned on their voice, which the worker serves like the base model. First done for Ritesh Sir on an M5 Pro (24 GB), 2026-10-26.

**Consent first.** The voice needs a consent record that covers training: `voicegw consents <voice_id>` should show a `fine_tune_attestation`. The source audio must be Arivihan's own.

## 1. Dataset

```bash
cd bench   # this env has voicegw, numpy, soundfile, openai
uv run python ../training/prepare_dataset.py LONG1.wav [LONG2.wav ...] \
    --reference ../gateway/data/voices/<voice_id>/reference.wav --out ../training/datasets/<name>
uv run python ../training/finalize_dataset.py ../training/datasets/<name>
```

- **`prepare_dataset.py` steps:**
  - Silero cuts the audio into 3–12 s clips that start and end on word gaps. A long stretch with no pause is cut at its quietest point.
  - Clips are dropped unless ECAPA rates them ≥ 0.55 similar to the reference, so other speakers fall out.
  - `gpt-transcribe` transcribes each clip, with brand names as a prompt.
  - `gpt-5.4-mini` puts English words into Latin script, matching how the gateway's LLM writes. The rewrite is rejected if its consonant skeleton drifts more than 0.2 from the original.
- **`finalize_dataset.py` steps:**
  - Clips whose transcripts contain digits are re-transcribed with numbers as spoken words, and dropped if digits remain.
  - A trailing "..." becomes ",".
  - Paths are made absolute.
  - About half the training rows get a `ref_audio`, because the worker clones from a reference.
- **Ritesh Sir's run:** a 26.6-min Arivihan video (`OUf5CklU7os`) plus a 1-min clean recording gave 136 train and 5 val clips, 23.5 minutes in total. Median speaker similarity was 0.886.

## 2. Train

```bash
cd training
# one-time: the official repo (OpenBMB/VoxCPM cce58a5) in its own env
SETUPTOOLS_SCM_PRETEND_VERSION=2.0.4.dev0 uv venv --python 3.11 voxcpm-src/.venv && \
  SETUPTOOLS_SCM_PRETEND_VERSION=2.0.4.dev0 uv pip install --python voxcpm-src/.venv -e voxcpm-src tensorboardX

# stop the worker first: training needs ~12-14 GB of unified memory
PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  voxcpm-src/.venv/bin/python train_voxcpm_ritesh.py --config_path conf/ritesh_lora_mps.yaml
```

- **The official trainer runs on MPS in fp32 unchanged.** CUDA autocast and the grad scaler turn themselves off, and torch.compile is skipped.
- **The watermark variables** make an out-of-memory condition fail fast instead of swapping the machine. The low watermark must be at or below the high one (the default is 1.4).
- **One patch:** `train_voxcpm_ritesh.py` is the official script plus one change, which runs the diffusion decoder only on audio rows. The loss is the same, with about 4× less decoder compute on text-heavy Hindi rows.
- **Config** (`conf/ritesh_lora_mps.yaml`): LoRA r=32, α=32 on the LM and DiT attention, lr 1e-4, batch 1 × accumulate 8, 400 steps, checkpoint and validation every 50.
  - Small datasets overfit within a few hundred steps, so the right checkpoint is chosen by evaluation, not by taking the last one.
- **Speed** on the M5 Pro: about 11 s per step, roughly 75 minutes for 400 steps.

## 3. Pick a checkpoint

```bash
voxcpm-src/.venv/bin/python eval_checkpoints.py ckpt/ritesh_r32 --ref-long REF30.wav --ref-short REF8.wav
# start the worker (it provides ECAPA), then:
cd ../bench && uv run python ../training/score_eval.py ../training/eval/ritesh_r32 --real ../training/datasets/ritesh/val_final.jsonl
```

- **What gets rendered:** three tutoring sentences the teacher never said, from the base model and every checkpoint, each with a long and a short reference.
- **Scores:**
  - Voice similarity: ECAPA against the held-out validation clips, which were never trained on.
  - Intelligibility: gpt-transcribe, measured as consonant-skeleton error.
  - Speed: real-time factor.
- **Choosing:** prefer the earliest checkpoint where similarity has plateaued without intelligibility dropping, then listen to it.

## 4. Serve

```bash
voxcpm-src/.venv/bin/python merge_lora.py <base snapshot dir> ckpt/ritesh_r32/step_XXXXXXX models/ritesh-voxcpm2
VG_WORKER_TTS_MODEL=$(pwd)/models/ritesh-voxcpm2 ...   # start the worker as usual
```

- **Why merge:** merging bakes the LoRA into the weights. The worker needs no code change and pays no extra per-step adapter cost.
- **The voice still clones from its reference.** The fine-tune moves the model toward the speaker's timbre, accent and delivery, so a short reference may now be enough. Measure both lengths in step 3.

## Results: Ritesh Sir, 2026-10-26

Voice similarity is ECAPA against held-out real clips, averaged over three unseen sentences. ASR error is the gpt-transcribe skeleton error (lower is better).

| Model | Reference | Voice similarity | ASR error | First chunk (worker) |
|---|---|---|---|---|
| base, zero-shot | 30 s | 0.692 | 0.156 | ~570 ms |
| step 250 | 30 s | 0.760 | 0.164 | 477 ms |
| **step 250 (serving)** | **8 s** | **0.748** | 0.156 | **349 ms** |

- **Checkpoint:** step 250 was the best on both reference lengths. Validation loss was noisy with 5 clips (best total at steps 100 and 350), and listening-style metrics decided it.
- **Speaking rate:** the fine-tune slowed delivery from 16.6 to 14.4 characters/s. The real Ritesh Sir speaks at 12.5 characters/s, so it moved toward his own pace.
- **Serving:** `training/models/ritesh-voxcpm2-step250` (merged, 5.4 GB) with the 8 s reference `datasets/ritesh/wavs/s00_0021.wav`. Start the worker with `VG_WORKER_TTS_MODEL` set to that path.
- **Limitation:** one worker serves one model, so every voice on this worker is rendered by Ritesh Sir's fine-tune. For several fine-tuned teachers, either run one worker per teacher, or load per-voice LoRAs at runtime (`load_lora` / `set_lora_enabled`, slower per step than merged weights).
- **Listening comparison:** `~/Desktop/ritesh_voice_ab/finetune/` (A = before, B = after).
