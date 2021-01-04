from Network_Costructor import Network_Constructor,Simp_Layer
from Neural_Network import Neural_Network

import numpy as np

test_net = Network_Constructor(act_f='sigm')
test_net + Simp_Layer(3,'none')
test_net + Simp_Layer(3)
test_net + Simp_Layer(1)

net = test_net.build()


data_inp = np.array([[1,1,1],
                     [1,1,0],
                     [1,0,1],
                     [1,0,0],

                     [0,1,1],
                     [0,0,1],
                     [0,1,0],
                     [0,0,0]])
data_out = [[1],
            [0],
            [0],
            [0],
            
            [0],
            [0],
            [0],
            [0]]

for epochs in range(10):
    for k in range(10):
        for pack in range(8):
            net.Feed_Forward(data_inp[pack])
            net.Back_Prop(data_out[pack],ln_speed=0.5)
    print(net.errors[-1])

for pack in range(8):
    net.Feed_Forward(data_out[pack])
    print(data_inp[pack])
    print(net.act_neurons[-1])