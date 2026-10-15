# AI lip reader

A Python research project for recognizing individual words from mouth movements in GRID videos. Work so far has progressed from a four-word color classifier to experiments covering 51 GRID words.

The current pipeline uses dataset alignment files to isolate words, crops the mouth in each frame, and trains a video classifier. It requires word boundaries supplied by the dataset. Full-sentence transcription and live video inference are future work.

## Work completed

### v0: four-word baseline

The first experiment classifies `blue`, `green`, `red`, and `white`. A frame-level CNN extracts visual features, and an LSTM combines them across the clip before classification.

The original project notes recorded these results as the number of training speakers increased:

| Training speakers | Recorded accuracy |
| --- | --- |
| 2 | 26% |
| 8 | 63% |
| 20 | 84% |

The 20-speaker result was considered a suitable baseline for expanding the vocabulary. These figures come from the existing notes; detailed run logs and configurations for those runs are not stored alongside them.

### v1: 51 GRID words

The second experiment extends clip extraction and classification to all non-silence GRID words. The results in [`results.txt`](src/ai_lip_reader/experiments/v1/results.txt) record:

| Model | Training duration | Recorded accuracy |
| --- | --- | --- |
| Baseline CNN + LSTM | 10 epochs | About 61% |
| ResNet CNN + bidirectional LSTM | 10 epochs | 72% |

The current v1 training code uses a 3D convolutional frontend, 2D residual blocks, and a two-layer bidirectional LSTM. It supports variable-length clips through batch padding and packed sequences. Training uses cross-entropy loss and AdamW, saves a checkpoint selected by the lowest average test loss, and displays a confusion matrix.

The script is currently configured for 25 epochs. The recorded results above describe earlier 10-epoch runs, not a verified result for the current configuration.

## Data pipeline

1. Read source `.mpg` videos and matching `.align` files.
2. Use alignment timestamps to extract word clips. v1 skips `sil` and `sp` and writes grayscale video through a three-channel MP4 format.
3. Detect facial landmarks with `face-alignment`, use mouth landmarks 48 through 67, and resize each mouth crop to 128 × 96 pixels.
4. Save each clip as a `uint8` PyTorch tensor with shape `[frames, 3, 96, 128]`.
5. Normalize pixels to `[0, 1]` when loading tensors for training.

Training assigns speakers `s21`, `s23`, `s24`, and `s25` to the test split when their data is present. Other speakers go into training. The v1 extraction and preprocessing scripts cover `s1` through `s25`, excluding `s21`. The speaker folders in the dataset must match the processed data available to the training script.

## Setup and usage

The project requires Python 3.12 or newer and uses `uv` for dependency management. OpenCV handles video processing, `face-alignment` detects landmarks, and PyTorch trains the models. PyTorch and NumPy are included in the lockfile as transitive dependencies.

From the repository root:

```sh
uv sync
```

Supply the GRID videos and alignment files locally in this layout:

```text
data/
├── videos/
│   └── s1/
│       └── <utterance>.mpg
└── alignments/
    └── s1/
        └── <utterance>.align
```

Repeat the speaker folders for the speakers being processed. The `data/` directory is excluded from Git, and the repository has no dataset download script.

Run the v1 pipeline in order:

```sh
uv run python src/ai_lip_reader/experiments/v1/create_clips.py
uv run python src/ai_lip_reader/experiments/v1/preprocessing.py
uv run python src/ai_lip_reader/experiments/v1/train.py
```

v1's `create_clips.py` extracts all non-silence words. It writes clips to `data/all_grid_word_clips/`; preprocessing writes tensors to `data/all_grid_word_processed/`, grouped by speaker and word.

Preprocessing currently hardcodes Apple's `mps` device. For a machine without MPS, edit the `FaceAlignment` device argument in `preprocessing.py` to use a supported device such as `cpu`. Training automatically chooses MPS when available and otherwise uses CPU.

Training writes `best_model.pth` and, for v1, `vocab.pkl` in the working directory. A training run overwrites an existing checkpoint with that filename. The repository includes a checkpoint, but does not record its model version or vocabulary metadata.

## Repository layout

```text
src/ai_lip_reader/
├── __init__.py          # Placeholder package entry point
└── experiments/
    ├── v0/              # Four-color-word extraction, cropping, and training
    └── v1/              # All-word extraction, cropping, training, and result notes
pyproject.toml           # Package metadata and dependencies
uv.lock                  # Locked dependency versions
best_model.pth           # Existing model checkpoint
```

The `ai-lip-reader` command currently prints a greeting. Use the experiment scripts to run the pipeline. v0's `test.py` is a scratch script, not a working inference tool or automated test suite.

## Future goals and development plans

These are proposed next steps, not completed features or measured results.

- Make experiments reproducible with saved configurations, random seeds, dataset counts, training logs, and model and vocabulary metadata in each checkpoint.
- Separate validation from final testing. Current training selects checkpoints using test loss; a dedicated validation split would leave held-out test speakers for final evaluation.
- Replace hardcoded paths, speaker lists, device choices, and training settings with configuration or command-line options.
- Improve preprocessing with crop-boundary checks, consistent color handling, and reports of skipped frames and failed clips.
- Add a standalone inference script that loads a checkpoint and vocabulary to classify a supplied word clip.
- Evaluate errors per word and per speaker, and compare model changes under the same data split and training budget.
- Extend recognition from isolated words to word sequences and full sentences, then investigate live video inference.
