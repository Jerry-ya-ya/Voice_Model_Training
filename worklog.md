# Voice_Model_Training

## 2026/09/22

- Connect project to remote Git repository.

## 2026/09/23

- Build and verify a complete modular PyTorch TTS MVP with LJSpeech support, checkpointed training, and HiFi-GAN inference.

- Document the CUDA-first Windows setup and GPU verification workflow.

## 2026/09/24

- Add a reproducible GPU benchmark for model throughput, memory, vocoder latency, and real-time factor.

- Add a reusable interactive TTS tester for synthesizing custom sentences from trained checkpoints.

- Add automatic checkpoint continuation with additional-epoch scheduling and preserved training state.

## 2026/09/25

- Add an interactive CLI menu for configurable new and continued TTS training runs.

- Add verified small, medium, and large LJSpeech training presets with configurable learning-rate decay.

- Redesign the training CLI with scanned arrow-key config selection and numbered per-config run directories.

## 2026/09/26

- Add persistent epoch timing with running averages and longest-epoch reporting to training.

- Route every CLI training and continuation launch to a new numbered run directory.
