<h1 align="center">
  <strong>AutoML for Breast Cancer Prediction: A Comparative Study of GWO and CG-GWO for Joint Feature Selection and SVM Hyperparameter Tuning</strong>
</h1>

<p align="center">
  <img src="https://i.pinimg.com/originals/be/00/47/be004788944ed7f9af035a47ff367b3b.jpg" width="700"/>
</p>

This repository contains a comprehensive Jupyter Notebook that performs a comparative study to optimize a Support Vector Machine (SVM) model using the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Repository (ID: 17).

The primary focus is comparing the performance of the **Original Grey Wolf Optimizer (GWO)** and the **Cauchy-Gaussian Grey Wolf Optimizer (CG-GWO)** for Joint Optimization (simultaneous Feature Selection and Hyperparameter Tuning).

## Methodology

1. **Data Acquisition & Preprocessing**: The dataset is fetched from the UCI Repository. Missing values are handled, features are scaled using `StandardScaler`, and the data is split using an 80:20 stratified split. An Exploratory Data Analysis (EDA) is performed to visualize class distributions and feature correlations.
2. **Baseline Model**: A default SVM (SVC) model is trained using all features to establish baseline performance metrics (Accuracy, F1-Score, and AUC).
3. **Joint Optimization Framework**: Simultaneous Feature Selection and Hyperparameter Tuning is achieved using a 32-dimensional search space (30 features, C, $\gamma$). A V4 V-shaped Transfer Function is used to binarize the continuous feature variables. The fitness function minimizes a weighted cost of the classification error rate and feature selection ratio.
4. **Optimization Execution**: We utilize the `mealpy` library to execute the Original GWO and a custom-implemented CG-GWO algorithm. Both algorithms are run for 30 independent runs to ensure statistical robustness.
5. **Interpretability & Visualizations**: The notebook generates convergence curves, confusion matrices, ROC curves, Precision-Recall (PR) curves, and feature selection frequency plots to interpret the results.
6. **Statistical Testing**: A Wilcoxon Signed-Rank Test is performed on the accuracies of the 30 independent runs to determine if the CG-GWO provides a statistically significant improvement over the standard GWO.

## Why CG-GWO?

The standard Original GWO algorithm, while effective, can sometimes suffer from premature convergence in high-dimensional spaces. The Cauchy-Gaussian Grey Wolf Optimizer (CG-GWO) addresses this by introducing mutations:
- **Cauchy Distribution**: Provides long tails, which allows the algorithm to make larger leaps in the search space. This greatly enhances global exploration and helps the algorithm escape local optima.
- **Gaussian Distribution**: Provides shorter jumps, which aids in local exploitation and fine-tuning around promising regions.
By combining these, CG-GWO achieves a superior balance between exploration and exploitation.

## Requirements

The required packages to run the notebook:
- `mealpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`
- `numpy`
- `scipy`
- `ucimlrepo`
- `jupyter`

## Results
The executed notebook (`executed_notebook.ipynb`) contains all the finalized tables, metrics, and visual plots comparing the Best C, Best Gamma, and selected features for both algorithms.
