# Problem Set 01 - Chest X-ray Classification

This project explores a pediatric chest X-ray dataset for a binary classification task:

- NORMAL
- PNEUMONIA

The dataset is stored in the following folder structure:

```text
data/
  chest_xray/
    train/
      NORMAL/
      PNEUMONIA/
    val/
      NORMAL/
      PNEUMONIA/
    test/
      NORMAL/
      PNEUMONIA/
```

## Project goals

- Inspect the dataset structure and class counts
- Check image dimensions and class distribution
- Visualize representative chest X-ray samples
- Preprocess images for a CNN
- Build a simple CNN for binary classification
- Train and evaluate the model on the provided splits

## Notebook

The main analysis notebook is:

- `Problem_Set_01_Dataset_Exploration.ipynb`

This notebook includes:

1. Dataset inspection
2. Counts by split and class
3. Image dimension analysis
4. Class distribution plots
5. Sample image visualization
6. Preprocessing pipelines for train/validation/test
7. Data augmentation for training images only
8. CNN architecture definition and summary
9. Stabilized CNN training with Adam, EarlyStopping, and ReduceLROnPlateau
10. Test-set classification report, confusion matrix, and ROC analysis
11. Interpretation of NORMAL and PNEUMONIA recall

## Environment

This project uses the virtual environment in the repository:

- `.venv\Scripts\python.exe`

The notebook expects to be run from the project root directory, which is:

- `D:\Problem Set 01`

## Important notes

- The original dataset files are not modified or deleted.
- The preprocessing is applied in TensorFlow pipelines rather than altering the source images.
- The notebook does not change the raw dataset folders.
- The test split is kept separate from training and validation and is used only for final evaluation.

## Training configuration

The CNN uses Adam with a learning rate of `0.0001`. Training includes:

- Early stopping with `restore_best_weights=True`
- Learning-rate reduction when validation loss plateaus
- A maximum of 30 epochs

The notebook evaluates the final model using a 0.5 probability threshold and reports precision, recall, F1 score, a confusion matrix, and ROC-AUC. Because missed pneumonia cases are false negatives, PNEUMONIA recall is treated as the most clinically important metric.

The latest recorded test-set results are:

- Test accuracy: `0.9054`
- NORMAL recall: `0.8248`
- PNEUMONIA recall: `0.9538`
- ROC-AUC: `0.9613`

## Typical workflow

1. Open the notebook in VS Code.
2. Select the project virtual environment.
3. Run the cells in order.
4. Review the dataset statistics and model results.

## Requirements

The project relies on Python packages such as:

- TensorFlow
- NumPy
- pandas
- matplotlib
- seaborn
- Pillow
- scikit-learn
