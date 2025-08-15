import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Preprocessing:
  def __init__(self, df: pd.DataFrame):
    self.df = df
    self.num_cols = None
    self.cat_cols = None

  def data_types(self):
    """Mengubah tipe data pada fitur record_date"""
    self.df['record_date'] = pd.to_datetime(self.df['record_date'])
    return self
    
  def handle_missing_duplicate_data(self, threshold: float = 50.0):
    """Menghapus kolom dengan missing >50% dan menghapus duplikat"""
    missing_value = self.df.isna().sum()
    missing_percentage = (missing_value / len(self.df)) * 100
    missing_col = missing_percentage.index.tolist()

    for col in missing_col:
      if missing_percentage[col] > 50:
        self.df = self.df.drop(columns=[col])

    self.df = self.df.drop_duplicates()
    return self

  def handle_outliers(self):
    """Menangani outlier dengan clip/log transform"""
    self.num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
    self.cat_cols = self.df.select_dtypes(include=['object']).columns

    for col in self.num_cols:
      Q1 = self.df[col].quantile(0.25)
      Q3 = self.df[col].quantile(0.75)
      IQR = Q3 - Q1

      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR

      outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
      outliers_percentage = (len(outliers) / len(self.df)) * 100

      if outliers_percentage > 0:
        if outliers_percentage < 5:
          self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)
        else:
          if (self.df[col] > 0).all():
            self.df[col] = np.log1p(self.df[col])

    return self

  def encode_df(self):
    """Encoding label & one-hot untuk fitur kategorikal"""
    self.df['has_loan'] = self.df['has_loan'].map({'Yes': 1, 'No': 0})
    self.df['education_level'] = self.df['education_level'].map({
            'Other': 0, 'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4
    })

    one_hot_features = ['gender', 'employment_status', 'job_title', 'region']
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    encoded = encoder.fit_transform(self.df[one_hot_features])

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(one_hot_features), index=self.df.index)
    self.df = pd.concat([self.df.drop(columns=one_hot_features), encoded_df], axis=1)

    return self

  def normalize_df(self):
    """Normalisasi data (Min-Max) tanpa kolom user_id"""
    user = self.df['user_id']
    self.df = self.df.drop(columns=['user_id', 'record_date'])

    for col in self.df.columns:
      min_val = self.df[col].min()
      max_val = self.df[col].max()
      self.df[col] = (self.df[col] - min_val) / (max_val - min_val)

    self.df['user_id'] = user  # kalau mau ditambahkan lagi
    return self

  def get_data(self):
    """Mengembalikan dataframe hasil preprocessing"""
    return self.df
  
if __name__ == "__main__":
  import os
  base_path = os.path.dirname(__file__)
  df_raw = pd.read_csv(os.path.join(base_path, '..', 'synthetic_personal_finance_dataset_raw.csv'))

  preprocessor = Preprocessing(df_raw)
  df_processed = (
    preprocessor \
    .handle_missing_duplicate_data(threshold=50) \
    .handle_outliers() \
    .encode_df() \
    .normalize_df() \
    .get_data()
  )

  df_processed.to_csv('preprocessing/dataset_preprocessing/synthetic_personal_finance_dataset_preprocessing.csv', index=False)
