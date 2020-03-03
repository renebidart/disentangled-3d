# disentangled-3d



## Process:

**DO MULTIPLE RUNS FOR EVERYTHING**
See how another paper implements it, like 5 runs, then use std + mean for graphs?

Try the 

#### February 26 - 29
**Hypothesis**: Because data is oriented somewhat consistently, it is harder to make a model encode randomly oriented data. Easier to encode a subset of the distribution.
* **Test**: MSE on rotation augmented data should be higher than non rotation augmented, given limited model capacity. Validation set will be augnemted the same way as training for both

**Hypothesis**: Optimizing rotation in the training loop will force the model to learn a subset of the distribution, improving performance (based on above assumption that its easier to learn a "subest of the distribution")
**Test**: 
* Training and validating using 90 rotation opt on augmented data should have lower mse
     * This was empirically true, but should run multiple tests for CIs
* The orientations fed into the model should converge to a single (or few / subset of all) during training, indicating the model gets better at encoding some than others. But it is possible it could be better at some than others at init too, so rather than convergence, we really just care about this subset.
    * Non-uniform orientation using the desk
    * Tests on other classes showed :?????
    * ?But this isn't a full disentangled representation. Maybe we just have to say we're close enough to a disentangled representation.
    


####  Before February 25
* Get ModelNet dataset, try plotting
* Make a 3D VAE 
    * Directly extend standard one I used for MNIST, based on https://arxiv.org/abs/1805.09190
    * Use standard pytorch affine / grid sampling functions like in spatial transformer net
* Dataloader with 90 degree rotation augmentation
* Forward pass testing all 90 degree angles:
    1. Try all 24 90 degree rotations in forward pass
    2. Record orientation with smallest loss, use this one (equivalently run another forward pass at this orientation)


## Formality:
Turn this all into proper Bayesian / VAE form:

Encode a conditional distribution, $p(x | orientation = \theta)$ instead of the full distribution, $p(x)$. If x and $\theta$ are independent, this is the same. We know they're different, and saw this both by looking and seeing visually they tend to be oriented in the same way, as well as the different performance when encoding a subset of the distribution.

How to talk about $p(x | orientation = \theta)$ being "simpler" than $p(x)$? What do I really mean?

During training we create (how exactly?) a model for $p(x | orientation = \theta)$. 
By testing $\theta$ (or optimizing over $\theta$), we find the orientation of a given sample. ? Choose $\theta$ to maximize the likelihood comes from this model?
??? Is this finding the $\theta$ that maximizes the posterior of ????


**Overall**
Model $p(x | orientation = \theta)$ instead of $p(x)$, and have a way to transform an x to this. Under what conditions is it possible to do this optimization? When there's a nice bijective transform? Can we optimize over every bijective transform instead of directly encoding?


---

## How to use?
* Download ModelNet10 dataset to ./data


## Rough Notebooks (Are NOT the same as the final versions of the models above):
* 101 - looking at modelnet dataset
* voxel_vae - process of making vae, 3d rotations, and all 90 rotation during training