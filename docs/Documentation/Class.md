---
layout: default
title: Class
parent: Documentation
---

# Class
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## PredictParameter

This class is used to convey the parameter required in LEG for model prediction. It contains four 

***Parameters***:

* \__init__(): initialization. Set category/opposite_category to -1 and value = 0; show_option=False.
* set_category(num): store the cetegory of the prediction probability. 
* value: the prediction probability within interval [0,1] .                                      
* set_opposite_category(num): if applied, the value represents P(category)-P(opposite_category).
* hide_show(): boolean which is used to show some partial result for debugging.










