import pandas as pd
import numpy as np


class DataContainer:
    def __init__(self, paths):
        assert isinstance(paths, dict), "Paths must be a dictionary with keys 'true' and 'fake' and values as the path to the data"
        assert all(isinstance(key, str) for key in paths.keys()), "Path to true data must be a string"
        
        self.paths = paths
        
        self.true = self.load_data(paths['true'])
        self.true['label'] = 1
        
        self.fake = self.load_data(paths['fake'])
        self.fake['label'] = 0
        
        self.feature = None
        self.label = None

        self.data = pd.concat([self.true, self.fake], axis=0).sample(frac=1).reset_index(drop=True)
        
    
    def load_data(sel, path: str):
        filetypes = ['clipboard', 'csv', 'excel', 'feather', 'fwf', 'gbq', 'hdf', 'html', 'json', 'orc','parquet', 'pickle',
                    'sas', 'spss', 'sql', 'sql_query', 'sql_table', 'stata','table','xls']
        
        assert path.split('.')[-1] in filetypes, "Filetype not supported, refer to pandas documentation for supported filetypes"
        return getattr(pd, f'read_{path.split(".")[-1]}')(path)
    
    def categorical(self, col: str):
        self.data[col] = self.data[col].astype('category')
    
    def date(self, col: str):
        self.data[col] = pd.to_datetime(self.data[col], format='mixed')
        
    def optimize_int(self):
        for col in self.data.select_dtypes(include=['int']).columns:
            self.data[col] = pd.to_numeric(self.data[col], downcast='integer')
    
    def optimize_float(self):
        for col in self.data.select_dtypes(include=['float']).columns:
            self.data[col] = pd.to_numeric(self.data[col], downcast='float')
    
    def feature_split(self, label: list, exclude: list):
        self.feature = self.data.drop([*label, *exclude], axis=1)
        self.label = self.data[label]
    def encode_categorical(self):
        for col in self.data.select_dtypes(include=['category']).columns:
            self.data[col] = self.data[col].map({val: num for num,val in enumerate(self.data[col].unique())}).astype('int')
            
    @property
    def head(self):
        return self.data.head()
    
    @property
    def tail(self):
        return self.data.tail()
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def info(self):
        return self.data.info(memory_usage='deep')
    
    @property
    def describe(self):
        return self.data.describe()
    
    @property
    def columns(self):
        return self.data.columns
    
    @property
    def dtypes(self):
        return self.data.dtypes
    