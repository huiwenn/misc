"Functions loading the .pkl version preprocessed data"
from tensorpack import dataflow
from glob import glob
import pickle
import os

class PedestrainPklLoader(dataflow.RNGDataFlow):
    def __init__(self, data_path: str, shuffle: bool=True, max_num=60):
        super(PedestrainPklLoader, self).__init__()
        self.data_path = data_path
        self.shuffle = shuffle
        self.max_num = max_num
        
    def __iter__(self):
        pkl_list = glob(os.path.join(self.data_path, '*'))
        pkl_list.sort()
        if self.shuffle:
            self.rng.shuffle(pkl_list)
            
        for pkl_path in pkl_list:
    
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            if sum(data['man_mask']) > self.max_num:
                continue
            if 'pos12' not in data.keys():
                continue
            yield data



def read_pkl_data(data_path: str, batch_size: int, 
                  shuffle: bool=False, repeat: bool=False, **kwargs):
    df = PedestrainPklLoader(data_path=data_path, shuffle=shuffle, **kwargs)
    if repeat:
        df = dataflow.RepeatedData(df, -1)
    df = dataflow.BatchData(df, batch_size=batch_size, use_list=True)
    df.reset_state()
    return df

