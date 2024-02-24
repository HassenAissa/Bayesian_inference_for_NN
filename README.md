# Beyesian_inference_for_NN

This is an implementation of Bayesian inference for neural networks using Tensorflow.

We have currently implemented the following:
- [x] SWAG
- [X] Bayes by Backprop (BBB)
- [X] SGLD
- [X] HMC
- [X] Support to Weights and Biases
- [X] Support to OpenAi Gym RL envirenment
- [X] Decision boundary visualisation
- [X] Uncertainty and performance metrics

## Setup:
You need to setup the package before running the library. In "Beyesian_inference_for_NN" directory run:
```
pip install -e .
```

## Requirements
Make a virtualenv:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the requirements:
```bash
pip install -r requirements.txt
```

## Usage
Unit test 1 (Classification):
```bash
make test1
```

Unit test 2 (Regression):
```bash
make test2
```

Gym (in progress):
```bash
make gym
```
