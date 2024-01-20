import os
import pandas as pd


class Preprocess:
    def __init__(self, DATA_DIR="data/official"):
        self.X_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col=0).drop(columns=["MIS_Status"])
        self.y_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col=0)["MIS_Status"]
        self.X_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col=0)

    def get_data(self, is_raw=True):
        return self.X_train.copy(),self.y_train, self.X_test.copy()
        
    def _str2float(self, s):
        return float(s)
    def _fill_year(self, s):
        return float("20"+s) if float(s) < 50 else float("19"+s) if float(s)>50 else s
    def _date_columns_process(self, date_columns):
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
        for datecol in date_columns:
            for i, (key, mapfunc) in enumerate(datemaps_dict.items()):
                self.X_train[datecol + "_" + key] = self.X_train[datecol].str.split("-").str[i].map(mapfunc)
                self.X_train[datecol + "_" + key] = self.X_train[datecol + "_" + key].astype(float)
                self.X_test[datecol + "_" + key] = self.X_test[datecol].str.split("-").str[i].map(mapfunc)
                self.X_test[datecol + "_" + key] = self.X_test[datecol + "_" + key].astype(float)
            self.X_train = self.X_train.drop(columns=[datecol])
            self.X_test = self.X_test.drop(columns=[datecol])
        return self

    def _money_columns_process(self, money_columns):
        for moneycol in money_columns:
            self.X_train[moneycol] = self.X_train[moneycol].str.replace("$", "").str.replace(",", "").astype(float)
            self.X_test[moneycol] = self.X_test[moneycol].str.replace("$", "").str.replace(",", "").astype(float)
        return self
    def general_preprocess(self, date_columns=["ApprovalDate", "DisbursementDate"], money_columns=["Term", "GrAppv", "SBA_Appv"]):
        self._date_columns_process(date_columns=date_columns)
        self._money_columns_process(money_columns=money_columns)
        return self
    
    def fit(self,):
        pass

    def transform(self,):
        pass

    def fit_transform(self,):
        self.fit()
        return self.transform()
    
    def data_augmentation(self,):
        pass