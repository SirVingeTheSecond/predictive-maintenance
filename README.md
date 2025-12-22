# Bearing Fault Diagnosis with Deep Learning

A deep learning project for diagnosing bearing faults using vibration signals from the Case Western Reserve University (CWRU) Bearing Dataset. This was developed as part of a Deep Learning course to explore how well neural networks can generalize to unseen fault severities.

## What this Project is about

Bearings are everywhere. Motors, turbines, conveyor belts, you name it. When they fail, things break down, and that costs money (and sometimes safety). The idea here is to use vibration signals to detect and classify bearing faults *before* catastrophic failure.

The CWRU dataset is the go to benchmark for this task. It contains vibration recordings from bearings with different fault types (ball, inner race, outer race) at different severity levels (0.007", 0.014", 0.021" fault diameter).

Most papers report 99%+ accuracy on this dataset. This seems to be largely due to data leakage. When you randomly split the data, samples from the same fault severity end up in both training and test sets. The model just memorizes severity specific patterns instead of learning what a "ball fault" actually looks like.

This project investigates that problem systematically and shows what happens when you use a more practical approach.

## Findings

Here is what has been found after running 90 experiments across different models, splits and classification modes:

| Split Strategy | CNN Accuracy | What It Tests |
|----------------|--------------|---------------|
| Random | 99.9% | Nothing useful (data leakage) |
| Cross Load | 99.4% | Generalization across operating loads |
| **Fault Size** | **64.2%** | Generalization to unseen severities |


**Other findings:**

* OR (outer race) fault completely fails with 0% recall on fault size split. The model literally cannot recognize OR faults it has not seen before.
* CNN beats LSTM and CNN LSTM on the realistic split. Simpler models generalize better when the task is hard.
* Ball fault is most robust and works well across all models (89 to 99% recall).
* GELU activation slightly outperforms ReLU** (67.4% vs 66.4%).

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

You will need PyTorch, NumPy, Matplotlib, Seaborn, scikit learn, and tqdm.

### 2. Get the Data

Download the CWRU bearing dataset and convert it to `.npz` format. Each file should contain a `DE` key with the drive end accelerometer signal.

Expected files in `data/raw/`:
```
1772_Normal.npz
1772_B_7_DE12.npz      # Ball fault, 0.007"
1772_B_14_DE12.npz     # Ball fault, 0.014"
1772_B_21_DE12.npz     # Ball fault, 0.021"
1772_IR_7_DE12.npz     # Inner race fault, 0.007"
1772_IR_14_DE12.npz    # Inner race fault, 0.014"
1772_IR_21_DE12.npz    # Inner race fault, 0.021"
1772_OR@6_7_DE12.npz   # Outer race fault, 0.007"
1772_OR@6_14_DE12.npz  # Outer race fault, 0.014"
1772_OR@6_21_DE12.npz  # Outer race fault, 0.021"
```

The same pattern for 1750 and 1730 loads applies.

The numbers (1772, 1750, 1730) represent motor RPM and load conditions where lower RPM means higher load.

## How to use

Everything goes through the CLI at `python -m src.main`. Here is the workflow:

### Quick Start: Run a Single Experiment

```bash
python -m src.main train --model cnn --split fault_size_all_loads --epochs 100
```

This trains a CNN on the fault size split (the realistic one) and prints the test metrics.

### Workflow: Run the Comparison Study

This is the main experiment. It trains CNN, LSTM, and CNN LSTM across all split strategies with 3 seeds each (90 total experiments):

```bash
# Step 1: Run all experiments (grab your popcorn)
python -m src.main run comparison

# Step 2: Analyze results
python -m src.main analyze comparison

# Step 3: Generate figures
python -m src.main figures comparison
```

### Commands

| Command | What it does                                                   |
|---------|----------------------------------------------------------------|
| `list` | Show available studies and sweeps                              |
| `run <study>` | Run all experiments in a study                                 |
| `status <study>` | Show completion progress                                       |
| `analyze <study>` | Print aggregated results with statistics                       |
| `figures <study>` | Generate all figures                                           |
| `export <study>` | Export results to CSV, Typst, or JSON (the purpose of this).   |
| `train` | Run a single specific experiment where you give the parameters |
| `test <checkpoint>` | Evaluate a saved model checkpoint                              |
| `sweep <sweep>` | Run a hyperparameter search                                    |
| `sweep results <sweep>` | Show the searches in a ranked order                            |

### Options for `train`

```bash
python -m src.main train \
    --model cnn \           # cnn, lstm, or cnnlstm
    --mode 4class \         # 4class, 10class, or multilabel
    --split fault_size \    # random, fault_size, fault_size_all_loads, cross_load
    --seed 42 \
    --epochs 100 \
    --force                 # overwrite existing results
```

### Options for `test`

```bash
# Evaluate on original split
python -m src.main test results/comparison/cnn_4class_fault_size_seed42/model.pth

# Evaluate on a different split
python -m src.main test results/.../model.pth --split random

# Compare across all splits
python -m src.main test results/.../model.pth --compare-splits
```

## Explanation of Splitting

This is the important part. The split strategy determines what goes in the training set versus the test set:

| Split | Training Severities | Test Severities | What It Tests |
|-------|---------------------|-----------------|---------------|
| `random` | All mixed | All mixed | Nothing (data leakage) |
| `fault_size` | 0.007", 0.021" | 0.014" | Interpolation to unseen severity |
| `fault_size_all_loads` | Same as above, all loads | Same | Most realistic |
| `cross_load` | Loads 0, 1 | Load 2 | Generalization across operating conditions |

**Use `fault_size_all_loads` for realistic evaluation.** The random split is only useful as a check.

## Classifications

| Mode | Classes | Task |
|------|---------|------|
| `4class` | Normal, Ball, IR, OR | Fault type classification |
| `10class` | Normal + 9 fault combinations | Fault type + severity |
| `multilabel` | Ball, IR, OR (binary each) | Multi label detection |

The paper focuses on `4class` since the main question is whether models can recognize fault types, not memorize severities.

## Other Experiments

Beyond the main comparison study, there are scripts for specific analyses:

### Testing Activation Functions

It seems like GELU and ReLU are the best performing although this has not been proven empirically. The paper uses GELU as standard.

For testing ReLU, Leaky ReLU, GELU, ELU, and SELU:

```bash
python experiments/run_activation_test.py
```

### Data Augmentation Sweep

Tests different augmentation implementations (noise, time warping, scaling):

```bash
python experiments/run_augmentation.py --mode quick   # test
python experiments/run_augmentation.py --mode full    # Full sweep
```

### Architecture and Dropout Sweep

Tests different dropout values across models:

```bash
python experiments/architecture_sweep.py quick   # Single seed
python experiments/architecture_sweep.py full    # All seeds
python experiments/architecture_sweep.py dropout cnn  # Focused on CNN
```

### OR Fault Root Cause Analysis

To illustrate an estimated guess on why OR fault fails to generalize:

```bash
python experiments/or_fault_analysis.py
```

This help visualize that OR fault severities do not follow a monotonic progression in feature space. The 0.014" severity looks completely different from 0.007" and 0.021", so my guess is that the model trained on such cannot recognize the middle.

## Understanding the Results

After running `python -m src.main analyze comparison`, this is the most likely output:

```
================================================================================
4CLASS_FAULT_SIZE (Fault size split, 4 class)
================================================================================
Model      Accuracy       Macro F1       Normal    Ball      IR        OR
--------------------------------------------------------------------------------
CNN        0.642+-0.015    0.608+-0.018    1.000     0.889     0.753     0.001
LSTM       0.505+-0.023    0.448+-0.031    1.000     0.995     0.125     0.000
CNNLSTM    0.554+-0.019    0.487+-0.025    1.000     0.994     0.313     0.000
```

The columns after Macro F1 are per class recalls. Notice how:

* Normal and Ball are easy (high recall)
* IR is moderate (CNN handles it, LSTM struggles)
* OR is a complete shitshow (0% recall across all models)

## Generated Figures

Running `python -m src.main figures comparison` creates these in `figures/comparison/`:

| Figure                            | Description |
|-----------------------------------|-------------|
| `fig01_model_comparison.png`      | Bar chart comparing model accuracies |
| `fig02_per_class_performance.png` | Per class recall |
| `fig03_split_comparison.png`      | Random vs fault size split |
| `fig04_confusion_matrix.png`      | Confusion matrices (counts + normalized) |
| `fig05_training_curves.png`       | Loss and accuracy over epochs |
| `fig06_roc_curves.png`            | ROC curves for multilabel classification |
| `fig07_tsne_features.png`         | t SNE visualization of learned features |
| `fig08_signal_examples.png`       | Raw signals and FFT spectra for each class |
| `fig09_activation_test.png`       | Activation function comparison |
| `fig10_summary_heatmap.png`       | Heatmap of all models across all splits |

All figures are saved as both PNG and PDF for convenience.

## Configuration

All hyperparameters and study definitions live in `src/config.py`:

```python
# Key settings
WINDOW_SIZE = 2048      # Samples per window
STRIDE = 512            # Window overlap
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
DROPOUT = 0.3
PATIENCE = 15           # Early stopping patience
```

To add a new study, just define it in the `STUDIES` dict in config.py and run it with `python -m src.main run <study_name>`.

## Future Work

Some ideas if you want to extend this:

1. **Domain adaptation** to train on one load, test on another, and use techniques like DANN to bridge the gap.
2. **Better OR fault handling** since OR might need a different feature representation or domain specific augmentation.
3. **Transformer architectures** because attention might help with the non monotonic severity patterns.
4. **Uncertainty quantification** to know when the model does not know.
