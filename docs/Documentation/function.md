---
layout: default
title: Function
parent: Documentation
mathjax: true
---

# Function
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## predict_XXX

 ***predict_XXX(ori_img, model, pred_paras)***

This function calculates the corresponding prediction values for specific model XXX. You can define yours here.  It includes: nomalize image data, get the category with largest probability if needed, calculating prediction probability.

***Parameters***:

* ori_img: original image data with shape (W,H,C) like (224,224,3).
* model: The model which includes predict() function.
* pred_paras: Class predictParameter, usually leave it since predict_XXX function will set the category to be the one with greatest prediction probability.                                     



## import_image

 ***import_image()***

Import image from a certain path, return a [n,w,h,c] matrix.

**Parameters**:

* simple parameters like the number, randomization, suffix , size of the image, show_option

## rgb2gray

 ***rgb2gray(rgb)***

change rgb images to grayscale.

**Parameters**:

* rgb: original image with rgb channels.
 
 

## leg_2d_deconv

 ***leg_2d_deconv(input_conv, size)***

deconvolution for image data with shape [a,b].

**Parameters**:

* input_conv: input array with shape [a,b].
* size: size of the deconvolution,from[a,b] to[a*size, b*size].

## leg_conv

 ***leg_conv(channel)***

generalization of function leg_2d_deconv. Can be applied on rgb image.

**Parameters**:

* channel: The channel which applied deconvolution operator, if -1, applied to all channels.
* model: The model which includes predict() function.
* pred_paras: Class predictParameter, usually leave it since predict_XXX function will set the category to be the one with greatest prediction probability.    

## create_sigma

 ***create_sigma(matrix_d)***

create the covariance matrix designed for the differencing matrix D.

**Parameters**:

* matrix_d: The differencing matrix D

## create_sparse_matrix_d

 ***create_sparse_matrix_d(p_size, padding)***

Generate the differencing matrix D with length of the image. 

**Parameters**:

* p_size: the length of image. It should not be big or the corresponding matrix d will run out of memory.
* padding: boolean to decide whether padding 0 on the border of the image.

## make_normal_noise

 ***make_normal_noise()***

Generate num_n normal perturbations with the covariance matrix sigma

**Parameters**:

None

##  LEG_perturbation

 ***LEG_perturbation(sample_method, gradient_scheme)***

Generate summation of f(x)x in the paper. 

**Parameters**:

* sample_method: whether to take random sample x1,x2,… or take symmetric sample like $$x_1, -x_1,   x_2, -x_2, …$$

* gradient_scheme: Weather to keep the sign of the perturbation like using sum $$f(x)x(default)$$ or $$\sum \lvert f(x) \lvert \lvert(x)\lvert$$


##  LEG_new_perturbation

 ***LEG_new_perturbation()***

obsolete

##  LEG_explainer

 ***LEG_explainer(conv_size, noise_lvl, lambda_arr, num_sample, method, penalty)***

Calculate LEG estimator 

**Parameters**:

* conv_size: how much you wish downsample the image.
* noise_lvl: noise level for the covariance matrix.
* lambda_arr: Lambda value for the TV penalty. Larger value gives more sparse solution.
* num_sample: number of the perturbations.
* method: only develop conv, don’t worry about it. New is not used at all.
* penalty: If None, LEG is computed; If 'TV', LEG-TV is computed.

##  leg_solver

 ***leg_solver(ds_mat, dy_mat, threshold)***

Solving the linear problem by mosek solver. You may need a certification to use it. There is free academic lisence that you can register. https://www.mosek.com/products/academic-licenses/

**Parameters**:

* ds_mat: $$d^+_t * \sigma$$
* dy_mat: pseudo $$D * \sum(f(x)x)$$
* threshold: $$\lambda_0$$ times the absolute maximum of dy_mat.

##  MISC

Other functions like get_mask, sensitivity_anal, generateHeatmap are used to generate evalution and visualization.


