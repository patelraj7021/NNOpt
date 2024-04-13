
# NNOpt - IN DEVELOPMENT

### Introduction and Motivation

Neural networks (NNs) are powerful tools that can produce accurate and reliable predictions for nearly any application when applied carefully. 
Of particular interest to regression analyses are Universal Approximation Theorems (UATs) - theorems that show for any given function, an appropriately configured feedforward fully-connected NN can approximate it to within some desired measure of error.
While early UATs (e.g., Cybenko, 1989) showed that the width of a NN can be increased, while holding depth as fixed, until the error measure is within the required threshold, recent work by Lu et al. (2017) proved a UAT for NNs with fixed width and arbitrarily high depth.
This fixed width, variable depth approach offers a significant advantage compared to the fixed depth, variable width approach in terms of the total number of nodes needed to achieve approximation, making the Lu et al. (2017) framework an attractive methodology for optimizing NNs for universal approximation.

Refer to the article for full details, but in summary, the Lu et al. (2017) framework proposes that a fully-connected *ReLU* NN with width $n+4$, where $n$ is the number of input dimensions, and arbitrarily high depth can approximate any Lebesgue-integrable function (a class of functions which are integrable but not necessarily smooth or real across all input dimensions). 
Due to the generalized nature of Lebesgue-integrable functions, many phenomena in the natural sciences fall under this class.
A practical, highly informative application is to compare the performance of these NN approximators with a chosen parametric function to determine if the function truly captures all information embedded in the input data.
That is, if the NN approximator is more accurate in predicting the target values than the parametric function, it shows the function's formulation can be improved upon for better performance. 
Conversely, if the NN approximator is similarly accurate to the parametric function, the function is formulated as well as possible for the given input data (in the context of predicting the chosen target values).

While libraries such as Keras, scikit-learn, PyTorch, etc. have made the generation of NNs accessible to anyone familiar with Python, optimizing hyperparameters for these NNs limits widespread use due to the commonly used "trial-and-error" process of hyperparameter choice. 
While quick and effective when the user is experienced with NNs, this process can be tedious and error-prone for users who are focused on mastering their respective fields as opposed to NN optimization.
This library is being developed for these users.
In its completed form, it will provide a fully automated method of hyperparameter optimization based on the Lu et al. (2017) framework - requiring users to only input features, target data, and a parametric function if they choose to compare against one.
It will output NN hyperparameters & performance, and a comparison with the parametric function across all input dimensions.

### High-Level Functionality Goals

1. For a regression problem for a given dataset, determine the neural network (NN) configuration that best fits the data
2. Compare performance (i.e., residuals) of the optimal NN with a parametric model (along each input dimension) and determine the level of similarity between the two
3. Create a SOM to fit a given dataset and compare against PCA; determine the level of non-linearity of the features


### Citations

Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals, and Systems (MCSS), 2(4):303–314.

Lu, Z., Pu, H., Wang, F., Hu, Z., and Wang, L. (2017). The expressive power of neural networks: A view from the width. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 6232–6240, Red Hook, NY, USA. Curran Associates Inc.
