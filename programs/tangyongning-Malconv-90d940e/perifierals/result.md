# CNN
## 1000
Epoch accuracy is 0.975, precision is 0.953, Recall is 1.000, F1 is 0.975.
## 750
Epoch accuracy is 0.938, precision is 0.908, Recall is 0.972, F1 is 0.938.
## 500
Epoch accuracy is 0.859, precision is 0.863, Recall is 0.962, F1 is 0.908.
## 250
Epoch accuracy is 1.000, precision is 1.000, Recall is 1.000, F1 is 1.000.
## Header only
Epoch accuracy is 0.984, precision is 0.975, Recall is 1.000, F1 is 0.987.
## Section only
Epoch accuracy is 1.000, precision is 1.000, Recall is 1.000, F1 is 1.000.

# Transformer
## 1000 8 head
Epoch accuracy is 0.850, precision is 0.864, Recall is 0.849, F1 is 0.850.
## 1000 4 head
Epoch accuracy is 0.938, precision is 0.916, Recall is 0.971, F1 is 0.941.
## 1000 16 head
Epoch accuracy is 0.863, precision is 0.886, Recall is 0.872, F1 is 0.869.
## Possibile reason 
Transformer对于局部特征的捕捉并不如CNN有效
CNN可以在较少的数据上更快地收敛, Transformer需要更多的数据和计算资源来训练

# Continual learning
## Normal Back propagation
1, Epoch accuracy is 1.000, precision is 1.000, Recall is 1.000, F1 is 1.000.
2, Epoch accuracy is 1.000, precision is 1.000, Recall is 1.000, F1 is 1.000.
3, Epoch accuracy is 0.906, precision is 0.933, Recall is 0.967, F1 is 0.949.
4, Epoch accuracy is 0.938, precision is 0.967, Recall is 0.967, F1 is 0.967.
5, Epoch accuracy is 0.938, precision is 0.928, Recall is 1.000, F1 is 0.963.
6, Epoch accuracy is 1.000, precision is 1.000, Recall is 1.000, F1 is 1.000.
7, Epoch accuracy is 0.969, precision is 1.000, Recall is 0.964, F1 is 0.981.
8, Epoch accuracy is 0.875, precision is 0.893, Recall is 0.962, F1 is 0.926.
9, Epoch accuracy is 0.906, precision is 0.904, Recall is 1.000, F1 is 0.949.
10, Epoch accuracy is 1.000, precision is 1.000, Recall is 1.000, F1 is 1.000.

Validation Accuracy: 0.8875

## CBP
1, Epoch accuracy is 0.938, precision is 0.969, Recall is 0.964, F1 is 0.965.
2, Epoch accuracy is 0.906, precision is 0.938, Recall is 0.964, F1 is 0.948.
3, Epoch accuracy is 1.000, precision is 1.000, Recall is 1.000, F1 is 1.000.
4, Epoch accuracy is 0.969, precision is 0.964, Recall is 1.000, F1 is 0.981.
5, Epoch accuracy is 0.938, precision is 0.967, Recall is 0.967, F1 is 0.967.
6, Epoch accuracy is 1.000, precision is 1.000, Recall is 1.000, F1 is 1.000.
7, Epoch accuracy is 0.969, precision is 0.964, Recall is 1.000, F1 is 0.981.
8, Epoch accuracy is 0.938, precision is 0.938, Recall is 1.000, F1 is 0.968.
9, Epoch accuracy is 1.000, precision is 1.000, Recall is 1.000, F1 is 1.000.
10, Epoch accuracy is 1.000, precision is 1.000, Recall is 1.000, F1 is 1.000.

Validation Accuracy: 0.9125