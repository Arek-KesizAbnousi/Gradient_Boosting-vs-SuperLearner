# Evaluation of Boosting Algorithms: XGBoost, LightGBM, AdaBoost, and SuperLearner

## Introduction

This project evaluates the classification accuracy of different boosting algorithms and an ensemble method similar to SuperLearner on the Sonar dataset. The evaluation is performed over 100 independent train/test splits to ensure robustness.

## Objectives

- **Compare the classification accuracy of the following boosting algorithms:**
  - XGBoost
  - LightGBM
  - AdaBoost

- **Evaluate an ensemble method similar to SuperLearner.**

- **Experimental Setup:**
  - Perform 100 independent train/test splits.
  - Measure and compare the accuracy of each model.

## Project Structure

The project is organized into the following components:

```plaintext
project/
├── data/
│   └── sonar_data.csv        # Dataset
├── main.py                   # Main script to run the experiments
├── boosting_models.py        # Boosting algorithms implementations
├── ensemble_model.py         # SuperLearner ensemble implementation
├── data_preprocessing.py     # Data loading and preprocessing functions
├── evaluation.py             # Model evaluation functions
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
### Languages and Libraries
- **Language:** Python 3.7 or higher
- **Libraries:**
  - **Data Manipulation:** `pandas`, `numpy`
  - **Machine Learning Models:** `scikit-learn`, `xgboost`, `lightgbm`
  - **Visualization:** `matplotlib`
    
## Data Loading and Preprocessing 

### Data Acquisition
The Sonar dataset is used for this analysis. It consists of 208 samples, each with 60 features representing sonar signal frequencies bounced off metal cylinders (mines) or rocks.
- Source:  [Sonar Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar%2C+mines+vs.+rocks))
  
### Preprocessing Steps
- Feature Scaling: Standardized the features to have zero mean and unit variance.
- Label Encoding: Converted class labels ('M' for Mine, 'R' for Rock) to binary format (1 for Mine, 0 for Rock).
- Train/Test Splits: Performed 100 independent train/test splits with a consistent test size for fair evaluation.
  

## Model Definitions
### Boosting Models (boosting_models.py)
Implemented the following boosting algorithms using their respective libraries:

- XGBoost: Utilizes gradient boosting with optimized computational speed and model performance.
- LightGBM: A gradient boosting framework that uses tree-based learning algorithms.
- AdaBoost: An ensemble learning method that combines weak classifiers to form a strong classifier.

### Ensemble Model (SuperLearner Equivalent) (ensemble_model.py)
- SuperLearner Equivalent: An ensemble method that combines predictions from the boosting models using a meta-learner (e.g., logistic regression) to improve overall performance.

## Training and Evaluation

### Evaluation Functions (evaluation.py)
- Metrics: Used classification accuracy as the primary evaluation metric.
- Cross-Validation: Averaged the results over 100 independent train/test splits to ensure robustness.
  
### Main Script (main.py) 
**Execution Flow:**
1.Load and preprocess the dataset.
2.Initialize models.
3.Train and evaluate each model over 100 splits.
4.Collect and save the results..

### Dependencies (requirements.txt)
All required packages in requirements.txt file.

```
pandas
numpy
scikit-learn
xgboost
lightgbm
```
### Results
 ** **
-XGBoost: 0.85
-LightGBM: 0.84
-AdaBoost: 0.82
-Ensemble Model: 0.87

Mean Accuracies over 100 iterations:

| Model                                   | Average Accuracy |
|-----------------------------------------|------------------|
| XGBoost                                 | **86.00%**       |
| LightGBM                                | **88.00%**       |
| AdaBoost                                | **89.00%**       |
| Ensemble Model (SuperLearner):          | **90.00%**       |
          
### Conclusion
The **ensemble model (SuperLearner)** achieved the highest average accuracy, indicating that combining multiple models can improve performance over individual algorithms.



## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Sonar Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar%2C+mines+vs.+rocks))

## License
This project is licensed under the MIT License. See the LICENSE file for details.


