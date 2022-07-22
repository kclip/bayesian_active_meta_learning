# bayesian_active_meta_learning
Code for paper "Bayesian Active Meta-Learning for Reliable and Efficient AI-Based Demodulation" 2022, https://arxiv.org/abs/2108.00785

The meta-learning using Hessian-vector-product is 

The following 

This repository includes two folders, one for each experiment considered.
 1) demodulation
  * main_demod.py: the main file, runs frequentist and Bayesian meta-learning
  * baml4demod.py: auxiliary file
 2) equalization 
  * main_eq_mtr.py: the main file for meta-training, save to mat files the learnt model parameters
  * main_eq_mte.py: the main file for meta-testing, loads from file system the model and meta-test
  * baml4eq.py: auxiliary file
  
In both folders, an empty sub folder named 'run' for files generated while running to be saved should be made.
