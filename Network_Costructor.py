import numpy as np

class Network_Constructor:
    def __init__(self,act_f='tanh',optim='grad'):
        self.act_f = act_f
        self.optim=optim
        self.main_container = []

    def __add__(self,new_layer):
        if new_layer.type=='simp':
            func = new_layer.act_f
            optim= new_layer.optim

            if func=='default': func=self.act_f
            if optim=='default':optim=self.optim

            lay = [new_layer.neurons,func,optim]
        self.main_container.append(lay)

class Simp_Layer:
    def __init__(self,lines,act_f='default',optim='default'):  
        self.type='simp'         
        self.act_f=act_f
        self.optim=optim
        self.neurons=np.random.random_sample(lines)
    
def get_test_net():
    test_net= Network_Constructor()
    test_net + Simp_Layer(3)
    test_net + Simp_Layer(2)
    test_net + Simp_Layer(3)

    for n in range(len(test_net.main_container[:])):
        print(test_net.main_container[n])

get_test_net()