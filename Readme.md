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

Python 3.13 is recommended on Windows. Torch and TorchAudio must use matching
versions. For an NVIDIA GPU, install their CUDA build first, then install the
remaining project dependencies.

Confirm that Windows can see the NVIDIA GPU:

```powershell
nvidia-smi
```

Create a fresh virtual environment:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
```

Install the CUDA 12.8 builds of PyTorch and TorchAudio:

```powershell
.venv\Scripts\python -m pip install `
  torch==2.8.0 `
  torchaudio==2.8.0 `
  --index-url https://download.pytorch.org/whl/cu128
```

Then install the remaining dependencies. The already installed CUDA packages
satisfy the pinned Torch requirements and will not be replaced:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

Verify that this exact virtual environment can use the GPU:

```powershell
.venv\Scripts\python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected output includes `CUDA available: True` and the NVIDIA GPU name. The
PyTorch version normally has a suffix such as `2.8.0+cu128`.

The default training configuration uses `device: auto`, which selects CUDA when
the check above is true and otherwise falls back to CPU. To prevent an
accidental CPU training run, set this in `configs/mvp.yaml`:

```yaml
training:
  device: cuda
```

With that setting, training stops with a clear error if CUDA is unavailable.
Linux users can follow the same ordering with `.venv/bin/python`. macOS does
not support CUDA and requires the CPU path or a future MPS implementation.

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

### Interactive CLI menu

The easiest way to start or continue training is the arrow-key menu:

```powershell
.venv\Scripts\python cli.py
```

The menu scans every `configs/*.yaml` and `configs/*.yml` file automatically.
Use the Up/Down arrow keys and Enter to choose a config; no config path needs to
be typed. Device and confirmation prompts also use arrow-key selection.

Choose **new training** to create the next positive run number inside the
selected config category. For example, the first three runs started from
`ljspeech_small.yaml` are organized as:

```text
runs/
`-- ljspeech_small/
    |-- 1/
    |-- 2/
    `-- 3/
```

Each numbered directory is a completely independent training run containing
its own checkpoints, logs, samples, plots, resolved config, and launch configs.

Choose **continue training** to first select the config category and then select
the numbered run to resume. The CLI restores that run's latest epoch checkpoint,
model, optimizer, scheduler, global step, and best validation loss, but writes
the continued training into the category's next unused positive run number. For
example, continuing run `1` creates run `2`; if runs `1` through `3` already
exist, either a new training or a continuation creates run `4`. The source run
is never overwritten. Each continuation config records its parent run number
and source checkpoint. For every continuation round you can choose additional
epochs, device, batch size, data/step limits, and whether to preserve or
explicitly reset the checkpoint learning rate.

Every confirmed launch saves the exact choices under the newly created numbered run:

```text
runs/<config_name>/<positive_run_number>/launch_configs/
```

Legacy flat runs such as `runs/ljspeech_mvp/` are left untouched and are not
automatically moved into numbered categories. They remain resumable with the
direct `--resume` command.

The direct commands below remain available for scripts and automation.

### LJSpeech model presets

Three production-shaped presets are included. They share the same 22.05 kHz,
80-bin HiFi-GAN-compatible Mel configuration, and expect the extracted dataset
at `data/LJSpeech-1.1`.

| Preset | Parameters | Hidden size | Encoder + decoder layers | Batch | Epochs | Intended use |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ljspeech_small.yaml` | 0.11 M | 64 | 1 + 1 | 4 | 60 | Lowest-load experiments and pipeline checks |
| `ljspeech_medium.yaml` | 0.83 M | 128 | 2 + 2 | 4 | 100 | Conservative everyday training |
| `ljspeech_large.yaml` | 6.42 M | 256 | 4 + 4 | 2 | 150 | Higher capacity with controlled batch memory |

These reduced presets write to `ljspeech_small_v2`, `ljspeech_medium_v2`, and
`ljspeech_large_v2` run directories. The separate names prevent incompatible
new architectures from overwriting checkpoints produced by earlier presets.

Start one directly:

```powershell
.venv\Scripts\python train.py --config configs/ljspeech_medium.yaml
```

Or launch `cli.py`, choose **new training**, and enter one of these files as
the base configuration. If CUDA runs out of memory on unusually long batches,
reduce `training.batch_size` first. Checkpoints are less frequent in the large
preset to avoid consuming excessive disk space.

All three presets were verified on an RTX 4070 SUPER with CUDA 12.8 using
forward and complete optimizer-step benchmarks at 1,000 Mel frames. Actual
memory use still varies with the longest utterance in each padded batch. GPU
compute utilization can still briefly reach 100% because training deliberately
uses available compute; these smaller presets primarily reduce work per step,
memory pressure, sustained power use, and total training time.

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
.venv\Scripts\python train.py --config configs/smoke_resume.yaml --resume runs/smoke_test/checkpoints/epoch_0001.pt
```

For normal continuation, let the trainer find the highest numbered checkpoint
and train a requested number of additional epochs automatically:

```powershell
.venv\Scripts\python train.py --config configs/mvp.yaml --continue-train 10
```

For example, if `epoch_0100.pt` is the latest checkpoint, this command restores
the model, optimizer, learning-rate scheduler, global step, prior metrics, and
best validation loss, then trains epochs 101 through 110. Omitting the number
adds one epoch:

```powershell
.venv\Scripts\python train.py --config configs/mvp.yaml --continue-train
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

## Interactive TTS tester

Start the interactive tester to load the model once and enter as many custom
English sentences as you want:

```powershell
.venv\Scripts\python tts_demo.py --device cuda
```

The tester automatically prefers `runs/ljspeech_mvp/checkpoints/best.pt` and
falls back to the smoke checkpoint when the trained model is unavailable. Use
`q`, `quit`, or `exit` to stop. Each sentence creates a timestamped WAV and Mel
plot under `demo_outputs/`.

Specify a checkpoint explicitly when needed:

```powershell
.venv\Scripts\python tts_demo.py `
  --checkpoint runs/ljspeech_mvp/checkpoints/best.pt `
  --device cuda `
  --vocoder hifigan
```

For a non-interactive single sentence:

```powershell
.venv\Scripts\python tts_demo.py `
  --text "Hello, this is my custom sentence." `
  --device cuda
```

Repeat `--text` to synthesize multiple sentences in one model-loading session.
The current character tokenizer supports English text; Chinese text requires a
future Chinese tokenizer, dataset, and retrained acoustic model.

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
learning rate, training time, validation time, and total measured epoch time.
During training, the console also shows the running average epoch time and the
slowest epoch so far. The final timing summary reports the wall-clock duration
of the current training session. Measured epoch time covers training and
validation; the session duration also includes plot, checkpoint, and sample
generation overhead. Older metrics files are upgraded automatically when a
continued run writes its first timed epoch. `plots/training.png` graphs the loss
and learning-rate histories. Checkpoints include the model, optimizer,
scheduler, configuration, epoch, and global step.

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

## Benchmark

Run a repeatable synthetic model benchmark without loading the dataset:

```powershell
.venv\Scripts\python benchmark.py --config configs/mvp.yaml --device cuda
```

This reports parameter count, forward latency and throughput, complete training
step latency, Mel frames per second, and peak CUDA memory. Defaults use a batch
size of 2, 64 text tokens, 256 Mel frames, 3 warmup iterations, and 10 measured
iterations. Override them when comparing hardware or configurations:

```powershell
.venv\Scripts\python benchmark.py `
  --config configs/mvp.yaml `
  --device cuda `
  --batch-size 8 `
  --token-length 100 `
  --mel-frames 400 `
  --warmup 5 `
  --iterations 50
```

Provide a trained checkpoint to additionally measure acoustic inference,
vocoder latency, end-to-end text-to-waveform latency, and real-time factor
(`RTF = synthesis seconds / generated audio seconds`):

```powershell
.venv\Scripts\python benchmark.py `
  --checkpoint runs/ljspeech_mvp/checkpoints/best.pt `
  --device cuda `
  --vocoder hifigan `
  --text "Hello, this is a text to speech benchmark."
```

An RTF below 1 means synthesis is faster than realtime. Each run also writes a
machine-readable timestamped report under `benchmarks/`; keep input dimensions
and iteration counts identical when comparing GPUs or code changes.

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
