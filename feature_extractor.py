from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, funcs):
        self.funcs = funcs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.groupby("file_flow_id").agg(self.funcs)
        return df.to_numpy()


if __name__ == '__main__':
    from scipy.stats import skew, kurtosis
    import numpy as np

    from fcs_reader import FCSReader
    fcs_reader = FCSReader(root_dir='raw_fcs')
    df = fcs_reader.fit_transform(
        X=['flowrepo_covid_EU_023_flow_001', 'flowrepo_covid_EU_035_flow_001',
           'flowrepo_covid_EU_004_flow_001', 'flowrepo_covid_EU_007_flow_001',
          ])
    print(df.shape)

    feature_extractor = FeatureExtractor(
        funcs=[
            np.mean, np.std, np.min, np.max, np.median, np.size,
            lambda x: np.quantile(x, 0.25), lambda x: np.quantile(x, 0.75),
            skew, kurtosis
        ]
    )

    X = feature_extractor.fit_transform(X=df)
    print(X.shape)