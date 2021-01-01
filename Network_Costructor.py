import numpy as np
from Neural_Network import Neural_Network

class Network_Constructor:
    def __init__(self,act_f='tanh',optim='grad',loss_f='mse'):
        self.__act_f  = act_f
        self.__optim  = optim
        self.__loss_f = loss_f

        self._param_container = []        

    def __add__(self,new_layer):
        if new_layer.type=='simp':
            func = new_layer.act_f
            optim= new_layer.optim

            if func =='default': func=self.__act_f
            if optim=='default':optim=self.__optim

            lay = [new_layer.lines,func,optim]
        self._param_container.append(lay)

    def build(self):
        _weights     = []
        _neurons     = []
        _act_neurons = []
        _errors      = []
        _act_f       = []

        if   self.__loss_f=='mse': loss = lambda x,y: np.sum((y-x)**2)/len(x)
        elif self.__loss_f=='mae': loss = lambda x,y: np.sum(np.abs(y-x))/len(x)

        lay = len(self._param_container)

        for s in range(lay):
            lin=self._param_container[s][0]

            if   self._param_container[s][1] == 'none': _act_f.append(lambda x: x)
            elif self._param_container[s][1] == 'tanh': _act_f.append(lambda x: (np.exp(2*x)-1)/(np.exp(2*x)+1))
            elif self._param_container[s][1] == 'sigm': _act_f.append(lambda x: 1/(1+np.exp(-x)))

            _neurons.append(np.zeros(lin))
            _act_neurons.append(np.zeros(lin))
            _errors.append(np.zeros(lin))

        for s in range(lay-1):
            lin=self._param_container[s][0]
            next_lin=self._param_container[s+1][0]
            
            _weights.append([])
            _weights[s]=np.random.random_sample((lin,next_lin))

        return Neural_Network(_weights,_neurons,_act_neurons,_errors,_act_f,loss)

class Simp_Layer:
    def __init__(self,lines,act_f='default',optim='default'):  
        self.type='simp'         
        self.lines=lines
        self.act_f=act_f
        self.optim=optim

class Another_Layer(Simp_Layer):
    def __init__(self,lines,act_f='default',optim='default'):
        super().__init__(lines,act_f='default',optim='default')