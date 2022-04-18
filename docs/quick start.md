---
layout: default
title: Quick Start
nav_order: 2
---

# Quick Start
{: .no_toc }

Before you begin, please read the [installation guide]({{ site.baseurl }}/docs/Installation) and have all required dependencies properly installed.

{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Main Structure

* `methods/LEGv0.py`: Implementation of LEG explainer.
* `ImageNetExp.py`: Create the 500 LEG and LEG-TV explanations on ImageNet.
* `Sanity/`: folder including implementations of cascading randomizations on LeNet-5 in MNIST and VGG-19 in ImageNet dataset.
* `Plots/ :` Implementations of plots in papers including sensitivity analysis.
* `table/keysizetable.py`:Implementation of computing key size for each saliency method.


## Usage

The `LEG_explainer` function is called with the following basic inputs:
```python
 LEG_explainer(inputs, model, predict_func, penalty, noise_lvl, lambda_arr, num_sample):
```
The function returns lists for all inputs. Each list contains the saliency map, original image, prediction of original image and corresponding lambda level for saliency map in turn. 

We also provide a customized function for visualization:
```python
generateHeatmap(image, heatmap, name, style, show_option, direction):
```
You can choose the "heatmap_only", "gray" or "overlay" style for the heatmap and decide whether display original saliency or its absolute value by the direction option.


## Example

Here is a quick demostration using the LEG to explain what the `VGG19 model` "sees" in the `trafficlight.jpg` when trying to classify it.

```python
##Import the required packages
from methods.LEGv0 import * 
if __name__ == "__main__":
    print("We are excuting LEG program", __name__)
    # read the image
    img = image.load_img('Image/trafficlight.jpg', target_size=(224,224))
    img = image.img_to_array(img).astype(int)
    image_input = np.expand_dims(img.copy(), axis = 0)
    image_input = preprocess_input(image_input)
    print("Image has been read successfully")
    # read the model
    VGG19_MODEL = VGG19(include_top = True)
    print("VGG19 has been imported successfully")
    # make the prediction of the image by the vgg19
    preds = VGG19_MODEL.predict(image_input)
    for pred_class in decode_predictions(preds)[0]:
        print(pred_class)
    chosen_class = np.argmax(preds)
    print("The Classfication Category is ", chosen_class)
    begin_time = time()
    LEG = LEG_explainer(np.expand_dims(img.copy(), axis = 0), VGG19_MODEL, predict_vgg19, num_sample = 10000, penalty=None)
    LEGTV = LEG_explainer(np.expand_dims(img.copy(), axis = 0), VGG19_MODEL, predict_vgg19, num_sample = 10000, penalty='TV', lambda_arr = [0.1, 0.3])
    end_time = time()
    plt.imshow(LEG[0][0], cmap='hot', interpolation="nearest")
    plt.show() #change the backend of matplotlib if it can not be displayed 
    plt.imshow(LEGTV[0][0], cmap='hot', interpolation="nearest")
    plt.show()
    
    print("The time spent on LEG explanation is ",round((end_time - begin_time)/60,2), "mins") 
```
