# Monocular Depth Estimation

This Project will estimate the depth of a monocular Image. In order to achive this we use a neural network with multiple hidden layers. 
We separate the neural network into two groups, the coarse layers and the refine layers.
Coarse layers will achive a very rough global depth quiet quickly, while refines will train for longer but achive very fine estimations.

In order to run this project create the /data folder and add the nyu-dataset. Run convert_mat_to_img. After that just run Task until you are satisfied with the result. (Note that training only creates checkpoints every few epoches, though this variable is changeable)

The structure is based on the paper found [here](https://arxiv.org/abs/1406.2283).

All of this is based on Tensorflow. 
CUDNN is higly recomanded.