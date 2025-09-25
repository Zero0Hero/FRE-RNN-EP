# Toward Practical Equilibrium Propagation: Brain-inspired Recurrent Neural Network with Feedback Regulation and Residual Connections


## Environments
```
python = 3.9.20
pytorch = 2.4.1
numpy = 1.26.4
torchvision = 0.19.1
matplotlib = 3.9.2
tqdm = 4.66.5
```

There are some jupyter notebook files, recording the data corresponding to some charts. Some cells are too long and may not display properly. The cell can be copied to other notepad softwares. Or directly run it, you will find a recording file under 'res_path' defined in the code, e.g., '.\res\First_symm_LN10_bsc_dynamics_LyapunovE_E-MNIST'. And in it, there are some figures (about training process), a text file names 'res.txt' (recording all final results and end time) and two folders about the error (gradient) and states of neurons. Taking the 'res.txt' as an example, it recorded:\
"
EP_L2_MNIST_First_symm_LN10_bsc_dynamics_LyapunovE_E-EP_b_sc0.001-EP_It2sta20- Time: 20250720-170253: train: 100.00%+-0.00% (100.00% 100.00%)	 test: 97.69%+-0.08% (97.58% 97.79%)
...

EP_L2_MNIST_First_symm_LN10_bsc_dynamics_LyapunovE_E-EP_b_sc0.01-EP_It2sta20- Time: 20250720-170605: train: 100.00%+-0.00% (100.00% 100.00%)	 test: 97.57%+-0.11% (97.41% 97.70%)
...
".

It indicates that the end time of first experiment (L2, b_sc0.001, i.e. 2-hidden-layers, $\beta_i=0.001$) is '20250720-170253'. And the end time of second experiment (L2, b_sc0.01, i.e. 2-hidden-layers, $\beta_i=0.01$) is '20250720-170605'. So the time cost of second experiment (5 repetitions) is 3min12s, i.e., one training session (50 epoch) cost 38.4s. 


## Table 1

'Layered_BP_EP_com_MNIST_512.ipynb': With layered architecture and Adam optimizer, the result of BP and EP.

'Layered_L3_bs20.ipynb': With layered architecture (3-hidden-layer), SGD, varying learning rates for all layers same as P-EP.

'Layered_L3_bs20_lrsame.ipynb': With layered architecture (3-hidden-layer), SGD, same learning rate for all layers.

'CNN_B/EP_com_MNIST.ipynb': With convolutinal architecture (32c-64c-10, same as the P-EP), the results of B/EP with feedback scaling $\beta_i=0.01, 0.1, 1$. 


## Table 2 and Table S1

'Layered_LN_(a)symm.ipynb': With layered architecture, Adam optimizer and (a)symmetric weights, the result of EP.

'Layered_L10_res/AGT/LN+_res.ipynb': With layered architecture, Adam optimizer and residual connections or arbitrary graph topologies (AGT), the result of EP.

'CNN_MNIST_comBP_c2fc1benchmark.ipynb': With convolutinal architecture (32c-64c-10), the results of B/EP. 

'CNN_CIFAR10_comBP_c3fc1benchmark.ipynb': With convolutinal architecture (32c-64c-128c-10), the results of B/EP. 


## Figure 4 

'Layered_ffsc_fbsc_2dplot.ipynb': The influence of feedforward scaling $α_i$ and feedback scaling $β_i$ on accuracy of MNIST classification. 


## Figure 5 

'Layered_L3bsc_Ctime_MLE_SR.ipynb' 


## Figure S8

'Layered_LN+_AGT_P.ipynb'


