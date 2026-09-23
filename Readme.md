# Voice Model Training MVP

An intentionally small, complete text-to-speech learning project. It trains a
non-autoregressive PyTorch acoustic model that maps normalized character tokens
to log-Mel spectrograms, then keeps waveform generation behind a replaceable
vocoder interface.

This is an educational baseline, not a production voice-cloning system.

## Architecture

```text
Raw text
   |
   v
Normalization -> Character tokenizer -> token IDs
                                      |
                                      v
                         Transformer encoder
                                      |
                         Length prediction and
                          sequence expansion
                                      |
                                      v
                         Transformer decoder
                                      |
                                      v
                            Mel spectrogram
                                      |
                         Pretrained HiFi-GAN
                                      |
                                      v
                                  waveform
```

Training uses the target Mel length to supervise sequence expansion. At
inference time the model predicts a frame-per-token ratio. This is simpler than
a full FastSpeech duration aligner while preserving a clean place to add one.

## Setup

Python 3.11 or 3.13 is recommended. Torch and TorchAudio are pinned together because
their binary versions must match.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

On Linux/macOS, activate or invoke the equivalent `.venv/bin/python`.

## Dataset and preprocessing

The default configuration downloads LJSpeech 1.1 automatically on first use
(about 2.6 GB extracted). Its rows are adapted to the project's common
`audio.wav|transcript` representation. Audio is loaded as mono, resampled to
22.05 kHz, converted to 80-bin natural-log Mel spectrograms, and cached under
`data/cache/`.

To add another single-speaker dataset, use `data.dataset: manifest` and provide
a pipe-delimited file:

```text
relative/or/absolute/audio.wav|transcript
```

Paths are relative to `data.root`. Dataset loading, normalization, tokenization,
Mel extraction, caching, splitting, and padding remain unchanged.

## Training

Full LJSpeech training:

```powershell
.venv\Scripts\python train.py --config configs/mvp.yaml
```

A tiny offline end-to-end smoke run:

```powershell
.venv\Scripts\python scripts/create_smoke_dataset.py
.venv\Scripts\python train.py --config configs/smoke.yaml
```

Resume by pointing at any saved checkpoint. Increase `training.epochs` beyond
the stored checkpoint epoch when resuming:

```powershell
.venv\Scripts\python train.py --config configs/smoke-resume.yaml --resume runs/smoke_test/checkpoints/epoch_0001.pt
```

Important hyperparameters, device selection (`auto`, `cpu`, or `cuda`), random
seed, sample cadence, and data limits live in YAML. CUDA is selected by `auto`
when available.

## Inference

```powershell
.venv\Scripts\python inference.py --checkpoint runs/ljspeech_mvp/checkpoints/best.pt --text "Hello, this is a speech synthesis test."
```

The default MVP uses TorchAudio's pretrained LJSpeech HiFi-GAN; its small
weight file downloads on first use. For offline pipeline checks, append
`--vocoder griffin_lim`. HiFi-GAN quality depends strongly on matching its
training-time Mel distribution, so early acoustic checkpoints will sound poor.

Inference writes:

```text
outputs/
|-- generated.wav
`-- generated_mel.png
```

## Experiment outputs

```text
runs/<experiment_name>/
|-- checkpoints/   # best.pt and periodic resumable snapshots
|-- samples/       # periodic reference-length Mel plots and WAV files
|-- plots/
|-- logs/
|   `-- metrics.csv
`-- config.yaml    # exact configuration copied into the run
```

Each metrics row records epoch, global step, training loss, validation loss,
and learning rate; `plots/training.png` graphs those histories. Checkpoints include the model, optimizer, scheduler,
configuration, epoch, and global step.

## Validation performed for this repository

The lightweight test suite, one-step CPU smoke training, checkpoint resume,
offline Griffin-Lim inference, and pretrained HiFi-GAN inference were executed
successfully. The generated artifacts are ignored by Git but remain under
`runs/smoke_test/` and `outputs/` in the working copy. A full LJSpeech download
and long training run were not executed as part of the smoke validation.

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```

Tests cover normalization/tokenization, manifest audio preprocessing and cache
creation, model shapes, loss calculation, and backward propagation.

## Known MVP limitations

- English character tokens only; there is no grapheme-to-phoneme system.
- Sequence expansion is interpolation supervised by utterance length, not
  phoneme-level duration alignment.
- No attention alignment, stop-token head, pitch, energy, or prosody controls.
- The small model/configuration is for learning the pipeline, not natural speech.
- HiFi-GAN is pretrained and fixed; the offline Griffin-Lim backend is only a
  validation fallback.
- Single-speaker only. Speaker embeddings, adaptation, emotional speech,
  real-time serving, and game-engine integration are deliberately out of scope.

The boundaries around tokenizer, dataset, acoustic model, and vocoder are kept
explicit so phonemes, a better duration model, speaker conditioning, and a
separately trained vocoder can be added later.
