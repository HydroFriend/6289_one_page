---
layout: default
title: Anaconda
parent: Installation
nav_order: 1
---

This page contains instructions to install LEG dependencies using [Anaconda](https://docs.anaconda.com/anaconda/install/). We strongly recommend importing a conda virtual environment with all the dependencies already satisfied from the `yml` file provided here as this would minimize the amount of time and headache to configure your own environment.
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Importing a conda venv from [LEG-tf.yml]({{ site.baseurl }}/download/LEG-tf.yml) file

Download the `LEG-tf.yml` file and open anaconda prompt to navigate to the location of the `LEG-tf.yml` file. Then type in the following command:

```bash
conda env create -f LEG-tf.yml
```
after this just place your `mosek.lic` file in the appropriate location then you are all set.

---

## Installing individual packages manually

Or if you prefer to install packages individually here are the required packages: 

* [cvxpy](https://github.com/cvxgrp/cvxpy) 
* [Mosek](https://www.mosek.com/documentation/)
* [Keras](https://www.tensorflow.org/install)
* [matplotlib](https://matplotlib.org/users/installing.html)
* [skimage](https://github.com/scikit-image/scikit-image)

and you would still just place your `mosek.lic` file in the appropriate location according to the mosek email.

