# Beyesian_inference_for_NN

This is an implementation of Bayesian inference for neural networks using Tensorflow.

We have currently implemented the following:
- [x] SWAG
- [X] Bayes by Backprop (BBB)
- [X] SGLD
- [X] HMC 
- [X] VADAM
- [X] Adam
- [X] SGD
- [X] BSAM

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
Install PyAce
```
pip install -e .
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
