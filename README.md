# Machine-Learning-Predictive-Modelling
List of various Machine Learning algorithms used in projects. 

## 1) [Predicting Car Prices using k nearest neighbors algorithm](https://github.com/SphericalSilver/Machine-Learning-Predictive-Modelling/blob/master/k%20nearest%20neighbors%20car%20price%20prediction.ipynb)

The dataset used was found [here](https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data).

The goal of the project was to determine which were the best features to use to predict a car's price, and then to see how accurate the predictions were using those models.

### Data Processing

The dataset went through an initial round of processing that included:

- adding column names
- changing the `type` of a column where necessary
- changing column values where necessary
- removing or filling missing values
- dropping unnecessary columns
- normalizing of numeric columns

### Univariate and Multivariate Modelling

Univariate Modelling was initially used to determine what the best features for modelling might be. This was done varying the k value (`n_neighbors`) hyperparameter in the `KNeighborsRegressor` function. 

![Univariate](https://i.gyazo.com/e332850a3ed67cbc5dd867922ea15587.png)

Thereafter, the best features were isolated, and multivariate modelling was performed using the 3 best, 4 best, and 5 best features, with the k value parameter varied from 1-25.

![Multivariate](https://i.gyazo.com/f7f51ed7e83a8fe005a8086b2af4bbc1.png)

It was determined that:

1. **Predictive modelling should be performed using the 4 or 5 best features** (the 5 best being, in order of best to worst, `['engine-size', 'width', 'horsepower', 'highway-mpg', 'curb-weight']`.
2. A **k-value of 3 or less** should be used to minimize the root mean squared error value.
