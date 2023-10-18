UCSL : Unsupervised Clustering driven by Supervised Learning
===============================

This repository contains codes and examples for UCSL : Unsupervised Clustering driven by Supervised Learning (UCSL) algorithm.

We aim at unsupervisedly identify subtypes and drive it with a supervised task (classification or regression).

The algorithm is implemented in a pedagogical, easy-to-use sklearn fashion with fit(), fit_predict(), predict() functions.

Method codes are available in "ucsl" directory, examples implemented in notebook are implemented in "examples" directory.

Installation
------------
Unless you already have sklearn, Numpy and Scipy installed, you need to install them:

```
sudo apt-get install python-numpy python-scipy
```

Clone the repository from github
```
git clone https://github.com/neurospin-projects/2021_rlouiset_ucsl.git
```

Add `UCSL` in your `$PYTHONPATH`


Original paper
-------

Please read and cite original UCSL paper : https://arxiv.org/abs/2107.01988
Cite as : 
Robin Louiset, Pietro Gori, Benoit Dufumier, Josselin Houenou, Antoine Grigis and Edouard Duchesnay. 
UCSL : A Machine Learning Expectation-Maximization framework for Unsupervised Clustering driven by Supervised Learning. 
ECML/PKDD, Sep 2021, Bilbao, Spain. 
