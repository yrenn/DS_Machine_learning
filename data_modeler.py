

from __future__ import annotations
import numpy as np
import pandas as pd 
from sklearn.ensemble import HistGradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle


class DataModeler:
    def __init__(self, sample_df: pd.DataFrame):
        '''
        Initialize the DataModeler as necessary.
        '''
        self.train_df = sample_df.copy()
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean'))
             #,('scaler', StandardScaler())
        ])

    def prepare_data(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        '''
        Prepare a dataframe so it contains only the columns to model and having suitable types.
        If the argument is None, work on the training data passed in the constructor.
        '''
        df = self.train_df if oos_df is None else oos_df
        df = df[['amount', 'transaction_date']]
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['transaction_date'] = df['transaction_date'].astype(np.int64)
        df['transaction_date'] = np.where(df['transaction_date'] == -9.223372036854776e+18, np.nan, df['transaction_date'])
        return df

    def impute_missing(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        '''
        Fill any missing values with the appropriate mean (average) value.
        If the argument is None, work on the training data passed in the constructor.
        '''
        oos_df[['amount']] = self.pipeline.fit_transform(oos_df[['amount']])
        oos_df[['transaction_date']] = self.pipeline.fit_transform(oos_df[['transaction_date']])
        return oos_df

    def fit(self) -> None:
        '''
        Fit the model of your choice on the training data paased in the constructor, assuming it has
        been prepared by the functions prepare_data and impute_missing
        '''
        X_train = self.impute_missing(self.prepare_data())
        y_train = self.train_df['outcome']
        # self.model = HistGradientBoostingClassifier(random_state=42)
        # self.model = RandomForestClassifier(n_estimators =100, random_state=42)
        self.model = DecisionTreeClassifier(random_state=42)
        self.model.fit(X_train, y_train)

    def model_summary(self) -> str:
        '''
        Create a short summary of the model you have fit.
        '''
        # return "enable_hist_gradient_boosting()"
        return f"Random Forest model."

    def predict(self, oos_df: pd.DataFrame = None) -> pd.Series[bool]:
        '''
        Make a set of predictions with your model. Assume the data has been prepared by the
        functions prepare_data and impute_missing.
        If the argument is None, work on the training data passed in the constructor.
        '''
        X_test = oos_df
        y_pred = self.model.predict(X_test)
        return pd.Series(y_pred)


    def save(self, path: str) -> None:
        '''
        Save the DataModeler so it can be re-used.
        '''
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> DataModeler:
        '''
        Reload the DataModeler from the saved state so it can be re-used.
        '''
        with open(path, 'rb') as f:
            return pickle.load(f)

transact_train_sample = pd.DataFrame(
    {
        "customer_id": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "amount": [1, 3, 12, 6, 0.5, 0.2, np.nan, 5, np.nan, 3],
        "transaction_date": [
            '2022-01-01',
            '2022-08-01',
            None,
            '2022-12-01',
            '2022-02-01',
            None,
            '2022-02-01',
            '2022-01-01',
            '2022-11-01',
            '2022-01-01'
        ],
        "outcome" : [False, True, True, True, False, False, True, True, True, False]
    }
)


print(f"Training sample:\n{transact_train_sample}\n")

# <Expected Output>
# Training sample:
#    customer_id  amount transaction_date  outcome
# 0           11     1.0       2022-01-01    False
# 1           12     3.0       2022-08-01     True
# 2           13    12.0             None     True
# 3           14     6.0       2022-12-01     True
# 4           15     0.5       2022-02-01    False
# 5           16     0.2             None    False
# 6           17     NaN       2022-02-01     True
# 7           18     5.0       2022-01-01     True
# 8           19     NaN       2022-11-01     True
# 9           20     3.0       2022-01-01    False


print(f"Current dtypes:\n{transact_train_sample.dtypes}\n")

# <Expected Output>
# Current dtypes:
# customer_id           int64
# amount              float64
# transaction_date     object
# outcome                bool
# dtype: object

transactions_modeler = DataModeler(transact_train_sample)

df=transactions_modeler.prepare_data()

print(f"Changed columns to:\n{df.dtypes}\n")

# <Expected Output>
# Changed columns to:
# amount              float64
# transaction_date    float64
# dtype: object

df_for_model=transactions_modeler.impute_missing(df)

print(f"Imputed missing as mean:\n{df_for_model}\n")

# <Expected Output>
# Imputed missing as mean:
#               amount  transaction_date
# customer_id
# 11            1.0000      1.640995e+18
# 12            3.0000      1.659312e+18
# 13           12.0000      1.650845e+18
# 14            6.0000      1.669853e+18
# 15            0.5000      1.643674e+18
# 16            0.2000      1.650845e+18
# 17            3.8375      1.643674e+18
# 18            5.0000      1.640995e+18
# 19            3.8375      1.667261e+18
# 20            3.0000      1.640995e+18


print("Fitting  model")
transactions_modeler.fit()

print(f"Fit model:\n{transactions_modeler.model_summary()}\n")

# <Expected Output>
# Fitting  model
# Fit model:
# <<< ANY SHORT SUMMARY OF THE MODEL YOU CHOSE >>>

in_sample_predictions = transactions_modeler.predict(df_for_model)
print(f"Predicted on training sample: {in_sample_predictions}\n")
print(f'Accuracy = {sum(in_sample_predictions ==  [False, True, True, True, False, False, True, True, True, False])/0.1}%')

# <Expected Output>
# Predicting on training sample [False  True  True  True False False True  True  True False]
# Accuracy = 100.0%

transactions_modeler.save("transact_modeler")
loaded_modeler = DataModeler.load("transact_modeler")

print(f"Loaded DataModeler sample df:\n{loaded_modeler.model_summary()}\n")

# <Expected Output>
# Loaded DataModeler sample df:
# <<< THE SUMMARY OF THE MODEL YOU CHOSE >>>

transact_test_sample = pd.DataFrame(
    {
        "customer_id": [21, 22, 23, 24, 25],
        "amount": [0.5, np.nan, 8, 3, 2],
        "transaction_date": [
            '2022-02-01',
            '2022-11-01',
            '2022-06-01',
            None,
            '2022-02-01'
        ]
    }
)

adjusted_test_sample = transactions_modeler.prepare_data(transact_test_sample)

print(f"Changed columns to:\n{adjusted_test_sample.dtypes}\n")

# <Expected Output>
# Changed columns to:
# amount              float64
# transaction_date    float64
# dtype: object

filled_test_sample = transactions_modeler.impute_missing(adjusted_test_sample)

print(f"Imputed missing as mean:\n{filled_test_sample}\n")

# <Expected Output>
# Imputed missing as mean:
#              amount  transaction_date
# customer_id
# 21           0.5000      1.643674e+18
# 22           3.8375      1.667261e+18
# 23           8.0000      1.654042e+18
# 24           3.0000      1.650845e+18
# 25           2.0000      1.643674e+18

oos_predictions = transactions_modeler.predict(filled_test_sample)
print(f"Predicted on out of sample data: {oos_predictions}\n")
print(f'Accuracy = {sum(oos_predictions == [False, True, True, False, False])/0.05}%')

# <Expected Output>
# Predicted on out of sample data: [False True True False False] ([0 1 1 0 0])
# Accuracy = 100.0%

 