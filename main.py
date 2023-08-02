from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectPercentile, mutual_info_classif, f_classif
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
from scipy.stats import skew, kurtosis
import numpy as np

from fcs_reader import FCSReader
from feature_extractor import FeatureExtractor
from plot_confusion_matrix import plot_confusion_matrix

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

file_flow_ids = [i.name for i in Path(r'raw_fcs').iterdir()]
eu_label = pd.read_excel('EU_label.xlsx')
labels = [
    eu_label.loc[eu_label['file_flow_id'] == file_flow_id, "label"].item()
    for file_flow_id in file_flow_ids
]
labels = [1 if i == 'Sick' else 0 for i in labels]

X_train, X_test, y_train, y_test = train_test_split(
    file_flow_ids, labels,
    train_size=0.6,
    random_state=RANDOM_STATE,
    stratify=labels
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(
    f"Train includes {sum(1 for i in y_train if i)} Sick and "
    f"{sum(1 for i in y_train if not i)} Healthy, "
    f"Test includes {sum(1 for i in y_test if i)} Sick and "
    f"{sum(1 for i in y_test if not i)} Healthy"
)


feat_union = FeatureUnion(
    transformer_list=[
        ('summary_encoder', FeatureExtractor(funcs=[
            np.mean, np.std, np.min, np.max, np.median, np.size,
            lambda x: np.quantile(x, 0.25), lambda x: np.quantile(x, 0.75),
            skew, kurtosis
        ]))
    ]
)


class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier=RandomForestClassifier()):
        self.classifier = classifier

    def fit(self, *args, **kwargs):
        self.classifier.fit(*args, **kwargs)
        self.__dict__.update(vars(self.classifier))
        return self

    def predict(self, *args, **kwargs):
        return self.classifier.predict(*args, **kwargs)


pipeline = Pipeline(
    steps=[
        ('read_data', FCSReader(root_dir='raw_fcs')),
        ('feat_union', feat_union),
        ('standard_scaler', StandardScaler()),
        ('select_percent', SelectPercentile()),
        ('clf', MyClassifier())
    ],
    verbose=True
)


params = {
    "select_percent__percentile": [25, 50, 75],
    "select_percent__score_func": [mutual_info_classif, f_classif],
    "clf__classifier": [
        RandomForestClassifier(random_state=RANDOM_STATE), 
        BaggingClassifier(random_state=RANDOM_STATE), 
        LogisticRegression(random_state=RANDOM_STATE)
    ],
}


gscv = GridSearchCV(
    estimator=pipeline,
    param_grid=params,
    scoring='f1',
    cv=3,
    verbose=4,
    n_jobs=-1,
    error_score='raise',
)

gscv.fit(X_train, y_train)

print(f"Best params:")
for i, j in gscv.best_params_.items():
    print(i, j)
print(f"Best CV score: {gscv.best_score_}")

y_pred = gscv.predict(X_test)

print("Evaluation on Test set:")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Sick'],))

cm = confusion_matrix(y_test, y_pred)


plot_confusion_matrix(
    cm=cm,
    normalize=False,
    target_names=['Healthy', 'Sick'],
    title="Confusion Matrix"
)