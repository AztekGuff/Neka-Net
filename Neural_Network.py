import numpy as np

class Neural_Network:
    def __init__(self,weights,delta_weights,neurons,act_neurons,errors,act_f):
        self.weights       = weights
        self.delta_weights = delta_weights
        self.neurons       = neurons
        self.act_neurons   = act_neurons
        self.errors        = errors
        self.act_f         = act_f

        self.layers = len(neurons)

    def Feed_Forward(self,inp):
        self.act_neurons[0] = inp

        for lay in range(1,self.layers):
            for lin in range(len(self.neurons[lay])):
                self.neurons[lay][lin] = np.sum(self.act_neurons[lay-1] *self.weights[lay-1][:][lin])
            self.act_neurons[lay] = self.act_f[lay][0](self.neurons[lay])

    def Back_Prop(self,target,ln_speed=0.1):
        self.errors[-1] = (target - self.act_neurons[-1]) *self.act_f[-1][1](self.neurons[-1])

        for lay in range(self.layers-2,0,-1):
            for lin in range(len(self.neurons[lay])):
                self.errors[lay][lin] = np.sum(self.errors[lay+1] *self.weights[lay][lin]) *self.act_f[lay][1](self.neurons[lay][lin])

        for lay in range(self.layers-1):
            for lin in range(len(self.neurons[lay])):
                for next_lin in range(len(self.neurons[lay+1])):
                    self.delta_weights[lay][lin][next_lin] = (self.neurons[lay][lin] *self.errors[lay+1][next_lin]) *ln_speed
            self.weights[lay] -= self.delta_weights[lay]
