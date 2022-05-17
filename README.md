# LAL-RL
This is a new version of Learning Active Learning which uses reinforcement learning.
This repository corresponds to the following paper: 
[Discovering General-Purpose Active Learning Strategies](https://arxiv.org/pdf/1810.04114.pdf) by Ksenia Konyushkova, Raphael Sznitman, and Pascal Fua.


Use 'build agent example.ipynb' to train an AL agent with reinforcement learning and use 
'test agent example.ipynb' to run an AL experiment to test an agent and other baselines. 
The agent will be stored in ./agents and the results of AL experiment will be stored in 
./AL_results.
To run the code, you will need numpy, sklearn, tensorflow, matplotlib, pickle packages. 
