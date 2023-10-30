# Windowed Non Linear Reparameterization
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)


This repository contains the code for Windowed Non Linear Reparameterization built on top of Blacjax. This work was done as a part of summer internship at Aalto University under the guidance of Dr. Nikolas Siccha and Prof. Aki Vehtari.

### Why:

At times MCMC algorithms have trouble sampling from distributions. One such example is Neal’s funnel in which due to strong non-linear dependence between latent variables. Non-centering the model removes this dependence, converting the funnel into a spherical Gaussian distribution.
