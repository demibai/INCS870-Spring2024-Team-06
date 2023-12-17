# INCS870-Fall2023-Team-03

## Overview
This project involves the development of a machine learning pipeline for network intrusion detection. The primary goal is to classify network activities into normal or attack categories, using the UNSW-NB15 dataset. The pipeline includes data preprocessing, feature engineering, model training, hyperparameter tuning, and model evaluation.

## Prerequisites
Before running this project, ensure you have the following installed:

- Python 3.x
- Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost
## Dataset
The dataset used is the UNSW-NB15, which can be downloaded from [https://research.unsw.edu.au/projects/unsw-nb15-dataset]. The dataset includes various features related to network traffic and a label indicating normal or attack activity.

## File Structure
- main.py: Main script containing the entire machine learning pipeline.
- data/: Directory containing the dataset files.
## Features
The project includes the following key components:

1. **Data Preprocessing**: Categorical data handling, feature scaling, and removal of highly correlated features.
2. **Dimensionality Reduction**: PCA is applied optionally to reduce feature dimensions.
3. **Model Training and Selection**: Several classifiers are trained, including Random Forest, Extra Trees, Logistic Regression, k-NN, Decision Tree, and XGBoost.
4. **Hyperparameter Tuning**: GridSearchCV is used for tuning model parameters.
5. **Model Evaluation**: Models are evaluated based on accuracy, precision, recall, and F1-score. Additional visualizations include ROC curves, precision-recall curves, and confusion matrices.
6. **Feature Importance Visualization**: For applicable models, feature importance is visualized.
## Usage
To run the project:

1. Place the dataset files in the data/ directory.
2. Run the main.py script: python main.py.
3. Optionally, modify the script to enable/disable PCA or change model parameters.
## Customization
You can customize the pipeline by:

- Modifying hyperparameters in the param_grid dictionary.
- Choosing different models or adding new ones in the models dictionary.
- Adjusting the PCA components in the preprocess_data_with_pca function.
## Visualization
The script includes code to generate various plots:

- Cumulative explained variance by PCA components.
- Feature importance for the best-performing model.
- Model performance comparison across different metrics.
- ROC curve and precision-recall curve for the best model.
## Contributions
Contributions to this project are welcome. Please ensure that you follow the existing code structure and comment on any significant changes.
## License
No license
