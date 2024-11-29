# main.py
#Set up a loop to perform 100 independent replications.
import numpy as np
from data_preprocessing import load_data, preprocess_data, get_train_test_split
from boosting_models import xgboost_model, lightgbm_model, adaboost_model
from ensemble_model import ensemble_model
from evaluation import evaluate_model

def main():
    data = load_data()
    X, y = preprocess_data(data)

    n_iterations = 100
    accuracy_table = np.zeros((n_iterations, 4))  # Columns: XGBoost, LightGBM, AdaBoost, Ensemble

    for i in range(n_iterations):
        X_train, X_test, y_train, y_test = get_train_test_split(X, y, train_size=158, random_state=i)

        # XGBoost
        model_xgb = xgboost_model()
        model_xgb.fit(X_train, y_train)
        accuracy_table[i, 0] = evaluate_model(model_xgb, X_test, y_test)

        # LightGBM
        model_lgbm = lightgbm_model()
        model_lgbm.fit(X_train, y_train)
        accuracy_table[i, 1] = evaluate_model(model_lgbm, X_test, y_test)

        # AdaBoost
        model_ada = adaboost_model()
        model_ada.fit(X_train, y_train)
        accuracy_table[i, 2] = evaluate_model(model_ada, X_test, y_test)

        # Ensemble Model
        model_ensemble = ensemble_model()
        model_ensemble.fit(X_train, y_train)
        accuracy_table[i, 3] = evaluate_model(model_ensemble, X_test, y_test)

    # Calculate mean accuracies
    mean_accuracies = np.mean(accuracy_table, axis=0)
    print("Mean Accuracies over 100 iterations:")
    print(f"XGBoost: {mean_accuracies[0]:.4f}")
    print(f"LightGBM: {mean_accuracies[1]:.4f}")
    print(f"AdaBoost: {mean_accuracies[2]:.4f}")
    print(f"Ensemble Model: {mean_accuracies[3]:.4f}")

if __name__ == "__main__":
    main()
