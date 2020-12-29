from Network_Costructor import Network_Constructor,Simp_Layer
from Neural_Network import Neural_Network

test_net = Network_Constructor()
test_net + Simp_Layer(3,'none')
test_net + Simp_Layer(2)
test_net + Simp_Layer(3)

net = test_net.build()

inp = [1.0,1.0,0.0]
net.Feed_Forward(inp)
