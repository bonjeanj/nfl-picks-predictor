from xgboost import XGBClassifier
def train_model(X,y):
 m=XGBClassifier(eval_metric="logloss");m.fit(X,y);return m
