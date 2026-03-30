# AlphaZero-Gomoku

A lightweight implementation of the AlphaZero algorithm for Gomoku (also known as Gobang or Five in a Row). This project provides a clear implementation to explore the core mechanics of AlphaZero, integrating Monte Carlo Tree Search (MCTS) with Deep Neural Networks.

By leveraging Gomoku as a strategic yet computationally manageable testbed, this implementation allows you to observe the development of strong tactical play from scratch through pure self-play reinforcement learning.

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **PyTorch**: Install the version matching your hardware (CUDA recommended for speed).
  ```
  # Example for standard install
  pip install torch torchvision torchaudio
  ```

### 2. Installation

Clone the repository, set up a virtual environment, and install the dependencies:

```
# Clone the repository
git clone <your-repo-url>
cd AlphaZero-Gomoku

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠 Training Workflow

This project follows a structured "Train-Plot-Archive" cycle to keep your experiments organized.

### Step 1: Run Training

The training script manages self-play, neural network updates, and evaluation. It automatically resumes from `checkpoints/latest.pth` if found. With no device flags, it uses CUDA when available, otherwise CPU.

```
python train.py [FLAGS]
```


| **Flag**                | **Description**                                                      |
| ----------------------- | -------------------------------------------------------------------- |
| `--use-gpu`             | Require CUDA (exit if unavailable).                                  |
| `--cpu`                 | Use CPU only, even if CUDA is available.                             |
| `--self-play-workers N` | Set number of parallel processes (default: `1`).                     |
| `--reset`               | Archive & Clear: Moves current logs/models to `runs/` and exits. |


### Step 2: Visualize Metrics

Generate plots showing Loss, Entropy, and Arena Win Rates (using Wilson Confidence Intervals).

```
# Plots the most recent CSV in logs/
python plot_metrics.py

# Specify paths; --z is the Z-score for Wilson CI (default 1.96 ≈ 95%; e.g. 2.576 ≈ 99%， 1.645 ≈ 90%)
python plot_metrics.py --input logs/metrics_6_6_4.csv --output my_plot.png --z 1.96

# Optional: display the figure instead of only saving PNG
python plot_metrics.py --input logs/metrics_6_6_4.csv --show
```

### Step 3: Archive Experiment

When you are ready for a new run, use the reset command to move all current outputs to a timestamped folder under `runs/archive_YYYYMMDD-HHMMSS/`.

```
python train.py --reset
```

---

## 🎮 Play Against the AI

Once trained, you can challenge the model using two different interfaces:

### 📺 Web UI (Recommended)

A modern, interactive board powered by Streamlit.

```
streamlit run UI_game.py
```

- **Note:** The UI scans the `results/` folder for available models. Please manually place the `.pth` files you wish to use into the `results/` directory before starting the app.

### ⌨️ Terminal CLI

A simple text-based interface for quick testing.

```
python human_play.py
```

- Input moves as `row,col` (e.g., `3,4`).

---

## 📂 Project Structure

- `checkpoints/`: Contains `latest.pth` and `best.pth` for specific board sizes.
- `logs/`:
  - `train.log`: Detailed execution text.
  - `metrics_<W>_<H>_<N>.csv`: Numerical data for plotting.
- `runs/`: The archive for completed experiments.
- `results/`: The directory used by the Web UI to load trained models.
- `*.pth`: Current and best policy weights are also mirrored in the project root for easy access.

> **Note:** Large `.pth` files and the `logs/` folder are excluded by `.gitignore`. If sharing models, please use external storage.

---

## 🔬 Technical Details

- **Board Configuration:** Default settings (e.g., 8×8 board, 5-in-a-row) are defined in the `TrainPipeline` class within `train.py`.
- **Model Consistency:** If you change dimensions in `train.py`, ensure you update the paths in `human_play.py` to match.

---

## 📚 References

1. *AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (Silver et al., 2017)
2. *AlphaGo Zero: Mastering the game of Go without human knowledge* (Silver et al., 2017)


