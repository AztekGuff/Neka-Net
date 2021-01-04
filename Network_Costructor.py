import numpy as np
from Neural_Network import Neural_Network

class Network_Constructor:
    def __init__(self,act_f='tanh',optim='grad'):
        self.__act_f  = act_f
        self.__optim  = optim

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
        _delta_weights = []
        _weights       = []
        _neurons       = []
        _act_neurons   = []
        _errors        = []
        _act_f         = []
    
        #'mse': loss = lambda targ,out: np.sum((out-targ)**2)/len(targ)
        #'mae': loss = lambda targ,out: np.sum(np.abs(out-targ))/len(targ)
        
        lay = len(self._param_container)

        for s in range(lay):
            lin=self._param_container[s][0]

            if   self._param_container[s][1] == 'none': _act_f.append([lambda x: x,                               lambda x: 0])
            elif self._param_container[s][1] == 'tanh': _act_f.append([lambda x: (np.exp(2*x)-1)/(np.exp(2*x)+1), lambda x: 1-x**2])
            elif self._param_container[s][1] == 'sigm': _act_f.append([lambda x: 1/(1+np.exp(-x)),                lambda x: x*(1-x)])

            tmp = np.zeros(lin)

            _neurons.append([])
            _neurons[s]=tmp

            _act_neurons.append([])
            _act_neurons[s]=tmp

            _errors.append([])
            _errors[s]=tmp

        for s in range(lay-1):
            lin=self._param_container[s][0]
            next_lin=self._param_container[s+1][0]
            
            _weights.append([])
            _weights[s]=np.random.random_sample((lin,next_lin))

            _delta_weights.append([])
            _delta_weights[s]=np.zeros((lin,next_lin))

        return Neural_Network(_weights,_delta_weights,_neurons,_act_neurons,_errors,_act_f)

class Simp_Layer:
    def __init__(self,lines,act_f='default',optim='default'):  
        self.type='simp'         
        self.lines=lines
        self.act_f=act_f
        self.optim=optim

class Another_Layer(Simp_Layer):
    def __init__(self,lines,act_f='default',optim='default'):
        super().__init__(lines,act_f='default',optim='default')