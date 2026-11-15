# onechess-tools

Testing and training tools for the [Onechess](https://github.com/ovnice-dev/onechess)
chess engine. The engine itself lives in its own repository; everything used to
measure, tune and train it is here.

## What is inside

| Folder | Content |
|---|---|
| `ci/` | SPRT and SPSA drivers for GitHub Actions (many machines, pentanomial statistics), weekly continuous-training pilot, Kaggle relay sessions |
| `tools/` | match arena (`arena.py`), SPSA tuner, NNUE trainers (`train.py` with PyTorch, `train_jax.py` with JAX for TPUs), network checker (`check_nnue.py`), binpack converters, full-threat feature indexer, gauntlet for rating estimates |
| `colab/` | scripts and notebooks for Kaggle and Google Colab (GPU and TPU training) |
| `lichess/` | setup of the Lichess bot ([OnechessBot](https://lichess.org/@/OnechessBot)): automatic updates from releases, tournament handling |
| `data/` | opening suites used for testing (UHO) |
| `reseau/` | experimental networks kept for reference |

Most comments and file names are in French.

## How Onechess is tested

- Every change to the engine is checked with a time-control SPRT (10+0.1,
  UHO openings, pentanomial statistics) run on GitHub Actions in waves of
  parallel games, before it is kept.
- Changes that should not alter play must keep the `bench 12` signature.
- Networks are trained on Kaggle GPUs (PyTorch) and TPUs (JAX), then verified
  bit for bit against the engine with `tools/check_nnue.py`.
- Search constants are tuned by SPSA at time control, then validated by SPRT.

## Training data

- Public Stockfish self-play data (`official-stockfish/master-binpacks` on
  Hugging Face, ODbL license).
- Binpack reading: code from `nnue-pytorch` (GPLv3), in `tools/binpack/lib/`.

## License

Copyright (C) 2026 ovnice. GNU General Public License version 3, like Onechess.
