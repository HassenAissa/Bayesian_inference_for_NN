# Installation
https://pypi.org/project/Pyesian/0.0.1/
# Introduction
Many successful pieces of software today involve machine learning, which is why many tools and libraries were created to make implementing machine learning models easier. The best examples are Tensorflow, PyTorch and Keras. We want good software to accelerate progress in ML. One area that has a lot of promise to accelerate progress is Bayesian deep learning.

Deterministic NNs seem to have many failure modes including discriminatory bias, poor uncertainty, and a lack of robustness. An important source of these is that the choice of the weights $\Omega$ of our neural network is often highly discriminating. $\Omega$ is often one local minimum among many others and does not encompass all the 'knowledge' needed to solve tasks. For instance, in a classification problem the choice of a particular $\Omega$ does not encode the uncertainty in the predictions being made. As long as our input point $x$ is 'far enough' from the decision boundary, the NN will predict class A with high probability, even if no data points exist around our input $x$. This can be seen on the illustrative plot below. 

An alternative that can potentially remedy the aforementioned failure modes is Bayesian learning. Bayesian machine learning involves inferring a posterior distribution $p(\Omega|D)$ over model parameters via Bayes rule given the dataset $D$: 
$$p(\Omega|D) = \frac{p(D|\Omega)p(\Omega)}{\int p(D|\Omega)p(\Omega)d\Omega}\$$
This rule has a prior $p(\Omega)$ which expresses prior knowledge and has been shown to have nice trustworthiness properties. Though extremely valuable to have, one cannot compute the exact posterior. Thus, approximate inference methods have been proposed. These diverse approaches to Bayesian inference, especially for deep learning, have not been united in a way that would enable Bayesian ML to have the same acceleration we have observed for deterministic ML. 

There exists many inference methods to approximate
posterior distribution (HMC, SGLD, SWAG, BBB...), each with its pros and cons (runtime, accuracy of approximation...). 

Our library builds on top of *Tensorflow Keras ©* to make the training of Bayesian Neural Networks simple. It groups together a wide range of inference methods in one easy-to-use framework to enable Bayes in ordinary NNs with minimum efforts on the user-side as well as many other features listed below.
# Inference Methods implemented
- [x] SWAG
- [X] Bayes by Backprop (BBB)
- [X] SGLD
- [X] HMC
- [X] SGD
- [X] SGLD
- [X] ADAM
- [X] VADAM
- [X] BSAM
# External libraries integrated
- [X] Support to all Keras models architectures
- [X] Support to all Tensorflow probability distributions
- [X] Support to Weights and Biases 
- [X] Support to OpenAi Gym RL envirenment
- [X] Tensorflow datasets easy loading
# Example codes
## Classification
```python
import sklearn
import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.nn import BayesianModel
from Pyesian.optimizers import BBB
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.visualisations import Metrics, Plotter

# Import dataset from sklearn
x,y = sklearn.datasets.make_moons(n_samples=2000)
# Wrap it in the Dataset class and indicate your loss
dataset = Dataset(
    tf.data.Dataset.from_tensor_slices((x, y)),
    tf.keras.losses.SparseCategoricalCrossentropy,
    "Classification"
)

# Create your tf.keras model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, activation='relu', input_shape=(2,)))
model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax))

# Create the Prior distribution
prior = GaussianPrior(0.0, -1.0)
# Indicate your hyperparameters
hyperparams = HyperParameters(lr=0.5, alpha=0.0, batch_size=1000)
# Instantiate your optimizer
optimizer = BBB()
# Provide the optimizer with the training data and training parameters
optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
optimizer.train(600)
# You're done ! Here is your trained BayesianModel !
bayesian_model: BayesianModel = optimizer.result()

# See your metrics and performance
metrics = Metrics(bayesian_model, dataset)
metrics.summary()
# Save your model to a folder
bayesian_model.store("bbb-saved")

# Visualize your results
plotter = Plotter(bayesian_model, dataset)
# Plot some distribution boundaries sampled from your posterior
plotter.plot_decision_boundaries(n_samples=100)
# Plot the uncertainty area of your model
plotter.plot_uncertainty_area(uncertainty_threshold=0.9)
```
## Regression
```python
import tensorflow as tf

from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import BBB, SGD
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.nn import BayesianModel
from Pyesian.visualisations import Metrics

# Create a dummy dataset
x = tf.random.uniform(shape=(600,1), minval=1, maxval=20, dtype=tf.float32)
y = 2*x+2
# Wrap it in the Dataset class and indicate your loss
dataset = Dataset(
    tf.data.Dataset.from_tensor_slices((x, y)),
    tf.keras.losses.MeanSquaredError,
    "Regression"
)

# Create your tf.keras model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, activation='linear', input_shape=(1,)))

# Indicate your hyperparameters
hyperparams = HyperParameters(lr=1e-3, frequency=1)
# Instantiate your optimizer
optimizer = SGD()
# Compile the optimizer with your data and the training parameters
optimizer.compile(hyperparams, model.to_json(), dataset, starting_model=model)
optimizer.train(2000)
# You are done! Here is your BayesianModel
bayesian_model: BayesianModel = optimizer.result()

# See your metrics and performance
metrics = Metrics(bayesian_model, dataset)
metrics.summary()
# Save your model to a folder
bayesian_model.store("sgd-saved")
```

# References 
- Tensorflow: https://www.tensorflow.org/
- Tensorflow probability: https://www.tensorflow.org/probability
- Tensorflow datasets: https://www.tensorflow.org/datasets
- Weights and Biases - The AI developer platform : https://wandb.ai/site
- Improving PILCO with Bayesian Neural Network Dynamics Models: https://www.cs.ox.ac.uk/people/yarin.gal/website/PDFs/DeepPILCO.pdf
- MCMC Using Hamiltonian Dynamics: https://www.mcmchandbook.net/HandbookChapter5.pdf
- Bayesian Learning via Stochastic Gradient Langevin Dynamics: https://www.stats.ox.ac.uk/ teh/research/compstat-s/WelTeh2011a.pdf
- A simple baseline for Bayesian uncertianty in deep learning https://arxiv.org/pdf/1902.02476.pdf
- Weight Uncertainty in Neural Networks https://arxiv.org/pdf/1505.05424.pdf
- Simple and Scalable Predictive Uncertainty Estimation using Deep Ensemble https://arxiv.org/pdf/1612.01474.pdf
- Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam: https://arxiv.org/pdf/1806.04854.pdf
- Sam as an optimal relaxation of Bayes: https://arxiv.org/pdf/2210.01620.pdf

