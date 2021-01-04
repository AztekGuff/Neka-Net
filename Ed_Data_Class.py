import numpy as np

class auto_encoder:
    def __init__(self,v_type,min_v,max_v):
        self.type=v_type
        self.min =min_v
        self.max =max_v
        
    def get_data_set(self,count):
        if   self.type=='int'  : return np.random.randint(self.min,self.max,count)
        elif self.type=='float': return np.random.uniform(self.min,self.max,count)