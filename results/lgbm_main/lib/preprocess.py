import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def get_data(DATA_DIR="data/official"):
    X_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col=0).drop(columns=["MIS_Status"])
    y_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col=0)["MIS_Status"]
    X_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col=0)
    return X_train, y_train, X_test


class Preprocess:
    def __init__(
            self, date_columns=["ApprovalDate", "DisbursementDate"],
            money_columns=["DisbursementGross", "GrAppv", "SBA_Appv"],
            categorical_columns=['RevLineCr', 'LowDoc', 'City', 'State', 'BankState']
    ):
        self.date_columns = date_columns
        self.money_columns = money_columns
        self.categorical_columns = categorical_columns
        
    def _str2float(self, s):
        return float(s)
    def _fill_year(self, s):
        return float("20"+s) if float(s) < 50 else float("19"+s) if float(s)>50 else s
    def _date_columns_process(self, X):
        month_map = {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12
        }
        datemaps_dict = {
            "day": self._str2float,
            "month": month_map,
            "year": self._fill_year,
        }
        for datecol in self.date_columns:
            for i, (key, mapfunc) in enumerate(datemaps_dict.items()):
                X[datecol + "_" + key] = X[datecol].str.split("-").str[i].map(mapfunc)
                X[datecol + "_" + key] = X[datecol + "_" + key].astype(float)
            X = X.drop(columns=[datecol])
        return X

    def _money_columns_process(self, X,):
        X[self.money_columns]= X[self.money_columns].applymap(lambda x: x.strip().replace('$', '').replace(',', '')).astype(float)
        return X

    def general_preprocess(self, X):
        X = self._date_columns_process(X)
        X = self._money_columns_process(X)
        return X
    
    def fit(self, X, y):
        self.oe_dict = dict()
        for col in self.categorical_columns:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            oe.fit(X.loc[:,col].values.reshape(-1, 1))
            self.oe_dict[col] = oe

    def transform(self, X_raw):
        X = X_raw.copy()
        X = self.general_preprocess(X)
        for col in self.categorical_columns:
            X.loc[:,col] = self.oe_dict[col].transform(X.loc[:,col].values.reshape(-1, 1))
            X[col] = X[col].astype(float)
        return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def data_augmentation(self, X, y):
        pass