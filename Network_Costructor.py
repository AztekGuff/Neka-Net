import numpy as np
from Neural_Network import Neural_Network

class Network_Constructor:
    def __init__(self,act_f='tanh',optim='grad'):
        self.__act_f = act_f
        self.__optim=optim
        self._param_container = []
        self._weights = []
        self._neurons = []
        self._act_neurons = []

    def __add__(self,new_layer):
        if new_layer.type=='simp':
            func = new_layer.act_f
            optim= new_layer.optim

            if func =='default': func=self.__act_f
            if optim=='default':optim=self.__optim

            lay = [new_layer.lines,func,optim]
        self._param_container.append(lay)

    def build(self):
        for s in range(len(self._param_container)):
            lin=self._param_container[s][0]

            self._neurons.append([])
            self._neurons[s] = np.zeros(lin)

            self._act_neurons.append([])
            self._act_neurons[s] = np.zeros(lin)

        for s in range(len(self._param_container)-1):
            lin=self._param_container[s][0]
            next_lin=self._param_container[s+1][0]
            
            self._weights.append([])
            self._weights[s]=np.random.random_sample((lin,next_lin))

        return Neural_Network(self._weights,self._neurons,self._act_neurons,self._param_container)

class Simp_Layer:
    def __init__(self,lines,act_f='default',optim='default'):  
        self.type='simp'         
        self.lines=lines
        self.act_f=act_f
        self.optim=optim

class Another_Layer(Simp_Layer):
    def __init__(self,lines,act_f='default',optim='default'):
        super().__init__(lines,act_f='default',optim='default')