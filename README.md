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

## 2) [Predicting House Prices using Linear Regression](https://github.com/SphericalSilver/Machine-Learning-Predictive-Modelling/blob/master/Linear%2BRegression%2B-%2BPredicting%2BHouse%2BPrices.ipynb)

The dataset can be found [here](https://dsserver-prod-resources-1.s3.amazonaws.com/235/AmesHousing.txt). This dataset was originally compiled by Dean De Cock for the primary purpose of having a high quality dataset for regression.

A pipeline of functions was set up to allow us to quickly iterate on different models, consisting of the `transform_features`, `select_features`, and `train_and_test` functions.

Feature Engineering, Feature Selection, and K-Fold cross validation were used on the original data-set to predict the price of houses in the city of Ames, Iowa, United States.

### Feature Engineering
 
- Any column with more than 15% missing values was dropped.
- For text columns in particular, we dropped cols with any missing values at all.
- For numeric columns, we imputed missing values as the average of that column.
- Created new features based on existing columns, such as no. of years until a house was sold.
- Dropped columns that weren't useful, or which leaked data on the sale.

### Feature Selection

- Identified numeric columns that correlated strongly with target columns, and selected those with strong correlations (> 0.4)
- Converted any remaining nominal features to categorical type.
- Generated Heatmap to identify collinearity between columns. 
![heatmap](https://i.gyazo.com/ad9c4e6e5fae91633fe67646ec689aaf.png)

### K-Fold Cross Validation

The `train_and_test` function was modified to accept a parameter k which controlled the type of cross validation that occured:

1. When k = 0 , holdout validation was be performed, which is what the function did by default.
2. When k = 1, simple cross validation (first time with train and test sets, second time with them swapped) was performed, and then the avg rmse was returned.
3. When k > 1, k-fold cross validation was performed using k number of folds.
