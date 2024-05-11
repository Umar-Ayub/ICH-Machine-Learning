# Model Evaluation Framework Documentation

## Overview

This project provides a comprehensive framework for evaluating different machine learning models on datasets related to Intracerebral Hemorrhage (ICH). The primary objective is to determine the best predictive models and techniques for outcomes like mortality and functional independence, using a variety of statistical metrics and machine learning algorithms.

### Files in This Project

- `model_evaluation.py`: Main script that conducts model evaluations using various machine learning models and techniques.
- `metrics.csv`: Generated CSV file containing evaluation metrics for each model configuration.

## model_evaluation.py

### Overview

`model_evaluation.py` processes clinical data and evaluates the effectiveness of various machine learning models in predicting ICH outcomes. It handles data preprocessing, feature selection, model training, and performance evaluation, outputting a comprehensive set of metrics for each model.

### How to Use

1. Ensure you have a CSV file with patient data, named according to the filepath specified in the script (e.g., `ich_data_w_scores_modified.csv`).
2. Configure the script with the desired parameters, models, and evaluation strategies in the `model_params` and `configs` variables.
3. Run the script using a Python interpreter. The script will:
   - Load and preprocess the data.
   - Evaluate different models under various configurations.
   - Output performance metrics and save them in a CSV file.

### Model Evaluation Configurations

The script can be configured to evaluate models under different conditions:
- **Imbalance Correction**: Techniques like SMOTE and undersampling to handle imbalanced datasets.
- **Dimension Reduction**: Techniques like PCA and Boruta for feature selection and dimensionality reduction.
- **Cost Sensitivity**: Adjusting model parameters to handle class imbalances.

### Metrics Calculated

The evaluation focuses on several key metrics:
- AUC Score
- Precision
- Recall
- F1 Score

Each metric helps in assessing the performance of a model from different perspectives, catering to the nuances of medical outcome prediction.

### Output

- `metrics.csv`: A CSV file that logs the evaluation results for each model, including metrics like AUC, precision, recall, and F1 score.
- Graphs: Saves feature importance plots and Boruta results as images for visual analysis.

## Installation Requirements

To run this script, you need Python and the following libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- xgboost
- imblearn
- boruta

Ensure these dependencies are installed before running the script.

## Contributing

Contributions to enhance the model evaluation framework are welcome. Please consider the following steps for contributing:
1. Fork the repository.
2. Make improvements or additions.
3. Submit a pull request with a clear description of the changes and improvements made.

This project is an open platform for researchers and developers interested in advancing predictive modeling in the medical domain, specifically for ICH outcomes.
