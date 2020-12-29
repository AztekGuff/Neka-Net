import numpy as np

class Neural_Network:
    def __init__(self,weights,neurons,act_neurons,params):
        self.weights = weights
        self.neurons =neurons
        self.act_neurons = act_neurons
        self.params = params
        # [lines,act_f,optim]
        
    def Feed_Forward(self,inp):
        self.neurons[0] = np.array(inp)
        self.act_neurons[0] = np.array(inp)

        for lay in range(1,len(self.neurons)):

            if   self.params[lay][1] == 'none': func=lambda x: x
            elif self.params[lay][1] == 'tanh': func=lambda x: (np.exp(2*x)-1)/(np.exp(2*x)+1)
            elif self.params[lay][1] == 'sigm': func=lambda x: 1/(1+np.exp(-x))

            self.neurons[lay] = np.dot(self.act_neurons[lay-1],self.weights[lay-1])         #Если существует ошибка - она тут
            self.act_neurons[lay] = func(self.neurons[lay])

    def BackProp(self):
        pass



