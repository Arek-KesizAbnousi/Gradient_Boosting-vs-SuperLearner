#Ensemble Model (SuperLearner Equivalent) (ensemble_model.py)
#Python does not have a direct equivalent of R's SuperLearner, but you can implement an ensemble using stacking or voting classifiers from scikit-learn.
# ensemble_model.py

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from models.boosting_models import xgboost_model, lightgbm_model, adaboost_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

def ensemble_model():
    estimators = [
        ('xgb', xgboost_model()),
        ('lgbm', lightgbm_model()),
        ('ada', adaboost_model()),
        ('lda', LinearDiscriminantAnalysis()),
        ('svm', SVC(probability=True))
    ]
    # Using Stacking Classifier as an example
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        stack_method='predict_proba'
    )
    return model
