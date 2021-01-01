import numpy as np

class Neural_Network:
    def __init__(self,weights,neurons,act_neurons,errors,act_f,loss_f):
        self.weights     = weights
        self.neurons     = neurons
        self.act_neurons = act_neurons
        self.errors      = errors
        self.act_f       = act_f
        self.loss_f      = loss_f

    def Feed_Forward(self,inp):
        self.neurons[0]     = np.array(inp)
        self.act_neurons[0] = np.array(inp)

        for lay in range(1,len(self.neurons)):
            self.neurons[lay] = np.dot(self.act_neurons[lay-1],self.weights[lay-1])         #Если существует ошибка - она тут
            self.act_neurons[lay] = self.act_f[lay](self.neurons[lay])

    def BackProp(self,appr_out):
        appr_out = np.array(appr_out)
        self.errors[-1]=self.loss_f(self.act_neurons[-1],appr_out)

        for lay in range(len(self.neurons)-1,0):
            self.errors[lay] = (self.errors[lay+1],self.weights[lay])



