# crnn-pytorch
Use pytorch to implement CRNN and CTCloss, and generate libtorch models

paper: [An End-to-End Trainable Neural Network for Image-based Sequence
Recognition and Its Application to Scene Text Recognition](https://arxiv.org/pdf/1507.05717v1.pdf)  

## changelist
* Modify the VGG model in the original paper to MobileNetV3
* Replace LSTM in the original paper with dynamic LSTM


These operations will save about 50% of the amount of parameters and more than 60% of the amount of calculation, while maintaining or exceeding the recognition effect of the original paper.