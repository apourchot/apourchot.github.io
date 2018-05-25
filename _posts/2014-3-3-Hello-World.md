---
layout: post
title: Sample efficiency in Evolution Strategies
---

Evolution strategies are a class of stochastic, derivative-free black-box optimization algorithms. In an evolution strategy, individuals that are meant to be scored are sampled from a multivariate gaussian distribution. The parameters of this distribution (the mean and the covariance matrix) are then, based on the fitness of the individuals in such way that the expected fitness of the individuals of the next generation are maximized. To sum it up in an equation this means that ES algorithms work or the parameters $\theta$ of the distribution.

----
****

