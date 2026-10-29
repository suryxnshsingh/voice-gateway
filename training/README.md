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

### On the GPU server (acel4, NVIDIA L4)

About 2 s per step, so 500 steps take about 20 minutes. Gaurav Sir and Surya Sir were trained here on 2026-10-30.

```bash
# from the Mac: the trainer, the official source (without its .venv) and the dataset
ssh acel4 'mkdir -p ~/vg-training/conf ~/vg-training/datasets'
rsync -a --exclude .venv voxcpm-src train_voxcpm_ritesh.py eval_checkpoints.py acel4:vg-training/
rsync -a conf/<name>_lora_cuda.yaml acel4:vg-training/conf/ && rsync -a datasets/<name> acel4:vg-training/datasets/
# on acel4: point the manifests at the server's paths, build the env once, then train
sed -i "s#/Users/suryxnshsingh/Developer/arivihan/voice-gateway/training/datasets/#/home/ubuntu/vg-training/datasets/#g" datasets/<name>/*_final.jsonl
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
  nohup voxcpm-src/.venv/bin/python train_voxcpm_ritesh.py --config_path conf/<name>_lora_cuda.yaml > train-<name>.log 2>&1 &
```

- **Memory:** the run takes about 8 GB with `frozen_bf16: true`. Without it, it takes 12.5 GB.
  - The serving worker (about 9.7 GB) and the other team's `~/voice-cloning` app (about 6.8 GB) don't leave room for both. Pause `~/voice-cloning` only after its owners agree, and restart it afterwards.
  - Restart command: `cd ~/voice-cloning && nohup .venv/bin/python3 .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 >> work/api.log 2>&1 &`.
- **`frozen_bf16`** is the second patch. Training mode upcasts every weight to fp32. VoxCPM2 is stored and served in bf16, so casting the frozen base weights back is exact, and it saves 4.6 GB. LoRA weights and the VAE stay fp32.
- **Render the eval sentences on acel4** (`eval_checkpoints.py`, about 4 s per sentence). Score them on the Mac, where a speaker-only worker (`VG_WORKER_LOAD=speaker`) provides ECAPA.

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

## Results: Gaurav Sir and Surya Sir, 2026-10-30

Both were trained on acel4 (see "On the GPU server") from three 15-minute physics lessons each, with the Ritesh config at 500 steps. Since 8f9b0a7 a fine-tuned voice is served as a LoRA adapter on the base model, so neither needs a merged model or its own worker.

| Teacher | Dataset | Checkpoint | Voice similarity, 30 s / 8 s ref | ASR error, 8 s ref |
|---|---|---|---|---|
| Gaurav Sir, base model | 191 clips, 31.4 min | base | 0.612 / 0.692 | 0.156 |
| **Gaurav Sir** | | **step 350** | **0.672 / 0.705** | **0.156** |
| Surya Sir (Suryabhan Singh), base model | 210 clips, 33.4 min | base | 0.696 / 0.692 | 0.156 |
| **Surya Sir** | | **step 250** | **0.703 / 0.751** | **0.148** |

- **Serving:**
  - Gaurav Sir: `voice_e2af57e08f300ad6`, adapter `gaurav_step350`, 8 s reference `datasets/gaurav/wavs/s00_0046.wav`.
  - Surya Sir: `voice_a6634d550542d3d2`, adapter `surya_step250`, 8 s reference `datasets/surya/wavs/s00_0031.wav`.
  - Both are on suryansh@arivihan.com, next to Ritesh Sir.
- **Through the live worker,** on one unseen sentence, similarity to a held-out real clip was:
  - Gaurav Sir: 0.683 fine-tuned, 0.639 base.
  - Surya Sir: 0.689 fine-tuned, 0.618 base.
- **The gains are smaller than Ritesh Sir's.** The base model was already closer to these two voices, and with three eval sentences per checkpoint the scores are noisy. Listen before relying on the choice: `~/Desktop/gaurav_voice_ab/` and `~/Desktop/surya_voice_ab/` (A = base, B = fine-tuned, plus a real clip).
- **Lesson audio is full of numbers,** so `finalize_dataset.py` dropped about a fifth of the clips for digits it could not respell (54 and 43).
- **Transcripts:** gpt-transcribe sometimes ends a sentence with "。". Replace it with "।" before training.
- **Downtime:** both runs needed `~/voice-cloning` paused, for about 75 minutes in all.
