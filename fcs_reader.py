from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import FlowCal


class FCSReader(BaseEstimator, TransformerMixin):
    FEAT_COL_NAME = [f'f_{i}' for i in range(35)]
    ID_COL_NAME = 'file_flow_id'

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        output_df = pd.DataFrame(columns=[self.ID_COL_NAME] + self.FEAT_COL_NAME)
        for file_flow_id in X:
            for file_path in Path(rf'{self.root_dir}/{file_flow_id}').glob(r'*.fcs'):
                data = FlowCal.io.FCSData(file_path.as_posix())
                df_ = pd.DataFrame(data, columns=self.FEAT_COL_NAME)
                df_[self.ID_COL_NAME] = file_flow_id
                output_df = pd.concat([output_df, df_], axis=0)
        return output_df


if __name__ == '__main__':
    fcs_reader = FCSReader(root_dir='raw_fcs')
    df = fcs_reader.fit_transform(
        X=['flowrepo_covid_EU_023_flow_001', 'flowrepo_covid_EU_035_flow_001',
           'flowrepo_covid_EU_004_flow_001', 'flowrepo_covid_EU_007_flow_001',
          ])
    print(df.shape)
