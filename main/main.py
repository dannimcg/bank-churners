import pandas as pd 
import pickle
import pipeline

filename = r'model\xgboost_model.pkl'
with open(filename, 'rb') as f:
    model = pickle.load(f)

data = pd.read_csv(r"data\BankChurners.csv").drop(columns = ['Unnamed: 21'])

pipe = pipeline.Model(data, model)
pipe.pre_process()
pipe.run_model()
print(pipe.data.head())