# Tikhonov Regularization Term
This library includes the implemention of regularization tikhonov terms that were published from 2010 until now (2020). The used library is pytorch and all what you need to run this code is Pytorch and Numpy.

# How use it?
There is an example of how to use this lib (Weight Decay):
```
from regularization import *

network = Network() # The nn.Module that you buildt it!
loss_function = nn.CrossEntropyLoss()

loss_function_with_regularization = WeightDecay(network, loss_function)

# The training procedure with loss_function_with_regularization.backward()
```

# Where are the implemented papers?
You can read and enjoy from our review paper which is published on ... . In Table 5, you can find the regularization term and their papers.

