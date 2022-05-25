import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class Model():
    def __init__(self, data, model):
        """
        data is a dataframe in the same format as BankChurners.csv
        model is a pretrained model
        """
        self.data = data
        print(self.data.head())
        self.data_var = self.data[['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
       'Income_Category', 'Card_Category', 'Months_on_book',
       'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 'Credit_Limit', 'Avg_Open_To_Buy',
       'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
       'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
       'Marital_Status']]
        self.model = model
        self.continuous_vars = ['Customer_Age','Dependent_count', 'Months_on_book',
       'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 'Credit_Limit',
       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
    def pre_process(self):
        self.replace_key = {'Income_Category' : {'Less than $40K' : 0,
                                    '$40K - $60K' : 1,
                                    '$60K - $80K' : 2,
                                    '$80K - $120K': 3,
                                    '$120K +': 4,
                                    'Unknown' : np.nan},
              'Card_Category' : {'Blue' : 0,
                                 'Silver': 1,
                                 'Gold' : 2,
                                 'Platinum': 3},
              'Education_Level' : {'Uneducated' : 0,
                                    'High School' : 1,
                                    'College' : 2,
                                    'Graduate' : 3,
                                    'Post-Graduate': 4,
                                    'Doctorate' : 5,
                                    'Unknown': np.nan},
              'Gender': {'F' : 1,
                         'M' : 0},
              'Marital_Status': {'Unknown': np.nan}}
        self.data_var = self.data_var.replace(self.replace_key)
        self.data_clean = pd.get_dummies(self.data_var,  dummy_na=False, columns=['Marital_Status'])
        self.minVec = self.data_clean[self.continuous_vars].min().copy()
        self.maxVec = self.data_clean[self.continuous_vars].max().copy()
        self.data_clean[self.continuous_vars] = (self.data_clean[self.continuous_vars]-self.minVec)/(self.maxVec-self.minVec)
    def read_csv(self):
        self.data = pd.read_csv(self.filepath).drop(columns = ['Unnamed: 21'])
    def encode_result(self, string):
        if string == 0:
            return False
        else:
            return True
    def run_model(self):
        self.y_pred = self.model.predict(self.data_clean)
        self.y_pred = [*map(self.encode_result, self.y_pred)]
        self.data['PRED'] = self.y_pred

    
