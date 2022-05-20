## Deep Mixture of Gaussians

This repo contains the code of attempts to learn an action conditioned probabilistic model for septic patients.

The motivation behind the approach is to develop a simulated environment for septic patients, which can then be used for Model Based RL to learn optimal treatment strategies or to evaluate a learned policy.

The current implementation is minimal and uses a GRU based RNN to parameterize the means, covariance matrices, and components of a Mixture of Gaussians. Then the model will output an instance of Pytorch MixtureSameFamily.
Learning can be done by evaluating and minimizing the log likelihood.
