---

## Contents

### `classify.py`
- Trains a classifier network on the MNIST dataset.
- The trained model is saved as `classifier.pth` in the `Data/` directory.

### `data.py`
- Downloads the MNIST dataset if not already available.
- Handles preprocessing, such as normalization and reshaping.
- Provides convenient data loaders for training and testing.

### `plot.py`
- Contains all plotting utilities used throughout the project.
- Supports visualizing:
  - Generated samples
  - Comparison figures for the paper
  - Debugging figures and animations (e.g., GIFs)

### `usebest.py`
- Stores the best hyperparameters discovered during experimentation.
- These are (generally) used as default values across the different generative methods for consistency and reproducibility.

---

## Usage

These scripts are imported and used internally by `main.py` and generative method modules in the `Code/` directory. They are not meant to be executed standalone.

---