# Charge Scheduling of an Energy Storage System with Model-Based Reinforcement Learning under Dynamic Electricity Pricing

This is the repository for a research project for my master's thesis from 2019 that addresses the problem of effectively scheduling the charging of an energy storage system (i.e. a battery) to minimize the electricity cost in scenarios with variable electricity prices. The implemented approach is specifically designed for systems with local electricity generation provided by a solar array. A Bayesian neural network is trained to learn a model of how different variables (e.g. electricity prices, local electricity demand and battery charge) behave over time and in response to given battery charge rate. The learned model is then used in combination with other mathematical models as a simulation environment to train a deterministic neural network to minimize the electricity cost by effectively scheduling the charging of the battery.

The implemented approach is based on the research paper "Learning and Policy Search in Stochastic Dynamical Systems with Bayesian Neural Networks" by Depeweg et al.

Execute `main.py` to perform the policy optimization with a previously trained state transition model.
Optionally, `main.py` can also be used to train a new state transition model (see comments) or to compare the trained policy with two other policies in a test simulation (see comments).

## Used Python Version and Packages
The sofware was developed and tested with the following Python version and packages:
*  Python 3.7.2
*  Numpy 1.16.2
*  Tensorflow 1.13.1
*  Tensorflow Probability 0.6.0

Update: These packages are unfortunately no longer available in the specified versions for the used Python version and cannot be installed out of the box with a `requirements.txt` file. The code should still work with newer versions of the packages, but it may require some adjustments.
