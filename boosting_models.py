# boosting_models.py

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def xgboost_model():
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    return model

def lightgbm_model():
    model = lgb.LGBMClassifier()
    return model

def adaboost_model():
    base_estimator = DecisionTreeClassifier(max_depth=1)
    model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50)
    return model
