![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 2 -- Introduction to Machine Learning (Day 4 ODISSEI summer school)

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.


----------------------------------------------------------------------------

This is the Document for today: https://edu.nl/7hk9f

Collaborative Document day 1: https://edu.nl/ehx87

Collaborative Document day 2: https://edu.nl/7hk9f


##  🫱🏽‍🫲🏻 Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.

 If you feel that the code of conduct is breached, please talk to one of the instructors (if the complaint is for one of the participants) or send an email to training@esciencecenter.nl (if the complaint is for one of the instructors).
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

### 🖥 Workshop website
https://esciencecenter-digital-skills.github.io/2025-06-18-ds-sklearn-python-odissei

### 🛠 Setup
https://esciencecenter-digital-skills.github.io/2025-06-18-ds-sklearn-python-odissei#setup

## 👩‍🏫👩‍💻🎓 Instructors

Malte Lüken, Flavio Hafner

## 🧑‍🙋 Helpers

Sarah Alidoost

## 🗓️ Agenda

| **Time** | **Content**                                                                                              |
| -------- | -------------------------------------------------------------------------------------------------------- | 
| 09:00    | Cross-validation framework                                                                               |
| 09:50    | Break                                                                                                    | 
| 10:00    | Bias-variance trade-off                                                                                  | 
| 10:15    | Validation and learning curves (no exercise M2.01, M2.02)                                                |
| 10:35    | Manual hyperparameter tuning                                                                             | 
| 11:00    | Break                                                                                                    |
| 11:10    | Tuning continued, including automated tuning; skipped quiz M3.01 and dropped "need for a validation set" | 
| 11:50    | Evaluation and Hyperparameter tuning                                                                     | 
| 12:10    | Break                                                                                                    | 
| 13:10    | Evaluation and Hyperparameter tuning, continued                                                          |
| 13:55    | Break                                                                                                    | 
| 14:05    | Hyperparameter tuning continued                                                                          |
| 14:30    | Final exercise: Try out learned skills on new dataset                                                    |
| 15:00    | Break                                                                                                    |
| 15:10    | Final exercise, continued                                                                                |
| 15:30    | Concluding remarks                                                                                       |
| 15:45    | Wrap-up and post-workshop survey                                                                         |
| 16:00    | Guest lecture                                                                                            |
| 17:00    | END                                                                                                      |

<!-- 
**Leftovers from yesterday**
<div class="row">
  <div class="col-md-12">
    <table class="table table-striped">
      <tr> <th>Time</th> <th>Content</th></tr>
      <tr> <td>14:40</td>  <td>Overfitting and underfitting </td> </tr>
      <tr> <td>14:55</td>  <td>Cross-validation </td> </tr>
      <tr> <td>15:10</td>  <td>Break</td> </tr>
      <tr> <td>15:20</td>  <td>Cross-validation (continued) </td> </tr>
      <tr> <td>15:35</td>  <td>Bias-vs-variance trade-off </td> </tr>
      <tr> <td>15:55</td>  <td>Wrap-up</td> </tr>
    </table>
  </div>
</div>

<div class="row">
  <div class="col-md-8">
    <table class="table table-striped">
      <tr> <th>Time</th> <th>Content</th></tr>
      <tr> <td>09:00</td>  <td>Welcome and recap</td> </tr>
      <tr> <td>09:15</td>  <td>Validation and learning curves</td> </tr>
      <tr> <td>10:00</td>  <td>Break</td> </tr>
      <tr> <td>10:10</td>  <td>Manual tuning</td> </tr>
      <tr> <td>10:40</td>  <td>Tuning with grid search</td> </tr>
      <tr> <td>11:00</td>  <td>Break</td> </tr>
      <tr> <td>11:10</td>  <td>Tuning with grid search (continued) </td> </tr>
      <tr> <td>12:00</td>  <td>Lunch Break</td> </tr>
      <tr> <td>13:00</td>  <td>Evaluating tuning results </td> </tr>
      <tr> <td>14:00</td>  <td>Break</td> </tr>
      <tr> <td>14:10</td>  <td>Try out learned skills on new dataset</td> </tr>
      <tr> <td>15:00</td>  <td>Break</td> </tr>
      <tr> <td>15:30</td>  <td>Concluding remarks</td> </tr>
      <tr> <td>15:45</td>  <td>Wrap-up & Post-workshop Survey</td> </tr>
      <tr> <td>16:00</td>  <td>Guest lecture</td> </tr>
      <tr> <td>17:00</td>  <td>END</td> </tr>
    </table>
  </div>
</div>
 -->


## 🔧 Exercises

## 🧠 Collaborative Notes

### Cross validation 

We are using a new dataset `california_housing`. Load it as:

```python

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame = True)

```
look at data (what `housing` returns). The object `housing` has two attributes (variables), we get them as:

```python
data, target = housing.data, housing.target
```

look at data as:

```python
data
target
```

The goal is to predict the value of houses. We want to train a model (a decision tree regressor) on the dataset. 
This is a regression task. Import the regressor as:

```python
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=42) 
```

We use `random_state=42` to control random number generation. Then we fit the model to data as:

```python
regressor.fit(data, target)
```

We use MAE to evaluate the performance of the model:

```python
from sklearn.metrics import mean_absolute_error

target_predicted = regressor.predict(data)
score = mean_absolute_error(target, target_predicted)
score
```

The score is low (samll number) meaning the model fits the data perfectly. The `score` here is called "train error". But we are interested in "test error". Test error depends on how data is split to train/test splits. In scikit-learn we can split the whole data to train/test using `ShuffleSplit`. Another startegy which is more common is `KFold`, see [ different cross-validation strategies](https://scikit-learn.org/stable/modules/cross_validation.html#k-fold).

We do cross validation with a helper function:

```python
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

# first create an object and define the number of splits, test data size, set seed, 
cv = ShuffleSplit(n_splits=40, test_size=0.3, random_state=0)

# call the function cross_validate
cv_results = cross_validate(regressor, data, target, cv=cv, scoring="neg_mean_absolute_error")
```

Then lets have a look at the results by making a dataframe:

```python
import pandas as pd

cv_results = pd.DataFrame(cv_results)
cv_results
```

In `cv_results`, each row is related to one split in the `cv`. 

Let's visualize the distribution of the error as:

```python
cv_results["test_score"].plot.hist()
```

To get other stats of the error, we can have a look at mean and standard deviation of the error:

```python
cv_results["test_score"].mean()
cv_results["test_score"].std()
```

Let's inspect the distribution of the target:

```python
target.plot.hist()
```


### Overfitting and underfitting -- Quiz M2.01

#### 1: A model that is underfitting:

- a) is too complex and thus highly flexible
- **b) is too constrained and thus limited by its expressivity**
- **c) often makes prediction errors, even on training samples**
- d) focuses too much on noisy details of the training set

Select all answers that apply

#### 2: A model that is overfitting:

- **a) is too complex and thus highly flexible**
- b) is too constrained and thus limited by its expressivity
- c) often makes prediction errors, even on training samples
- **d) focuses too much on noisy details of the training set**

Select all answers that apply


#### 1: answer 

 b and c
 
#### 2: answer

 d and a
 
 
### 10 min break 

 :coffee: 
 
 
 ### Bias and Variance
 
 #### Resampling the training set
 
 What is the impact of re-drawing a new training set on the prediction error. Here we introduce two concepts:
 
 - Overfit: variance: when we use a more complex function, with every sample data, the fitted function changes depending of the data (and the noise within the that sample). on average, the predictions are not necessarily off, but each tend to fall far from the target.
 - Underfit: Bias: when we use less flexible function,the predictions cannot be on target on average.

#### The bias-variance decomposition of the Mean Squared Error (MSE)

For people with a background in mathematics and statistics, see https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff

#### Take home messages

- High bias == underfitting
- High variance == overfitting

*** The bias can come from the choice of the model family.


### Quiz M2.03

1. A model with a high bias:

a) is a characteristic of an underfitted model?

b) is a characteristic of an overfitted model?

c) when trained, exhibits greater sensivity to random resampling of the training data?

d) causes the learned prediction function to make systematic errors?

Select all answers that apply


2. A model with high variance:

a) is a characteristic of an underfitted model?

b) is a characteristic of an overfitted model?

c) when trained, exhibits greater sensivity to random resampling of the training data?

d) causes the learned prediction function to make systematic errors?

Select all answers that apply 


Answers:

1. a and d
2. b and c


### Comparing train and test errors

From a practical standpoint, how do we know on which side of the balance our model is sitting?

- Varying complexity: validation curves
- Varying the sample size: learning curves



#### Train vs test error: increasing complexity

We can look at the errors while varying the model complexity. The big picture is that models that are too simple have similar train and test error, while models that are too complex have a small train error but a very large test error. There is a sweet spot in the middle, and this is where good machine-learning models lie.

#### Varying sample size

Another useful way to look at the tradeoff between underfit and overfit is with varying sample size. 

#### Bayes error rate

 The error of the best model trained on unlimited data. Predictions are limited by noise (noisy data).
 
#### Model families

Crucial to match: statistical model and data-generating process 

*** Different model families: Simple variant vs Complex variant. 

*** forcing a model towards a simpler fit is called "regularization".


#### Take home messages

Models overfit: 
- number of examples in the training set is too small
- testing error is much bigger than training error 

Models underfit:
- models fail to capture the shape of the training set
- even the training error is large 


### Manual hyperparameter tuning: Set and get hyperparameters in scikit-learn

We can get and set the value of a hyperparameter in a scikit-learn estimator. Let's see how:

Load adult census dataset:

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

target_name = "class"
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
```

```python
target = adult_census[target_name]
data = adult_census[numerical_columns]
```

Inspect data:
```python
data.head()
```

Let's create a simple predictive model:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
```

We define the pipeline as:

```python
model = Pipeline(
    steps=[
        ("preprocessor", StandardScaler()),
        ("classifier", LogisticRegression()),
    ]
)
```

Now, we can use the `model` to access the hyperparameters. 

We can evaluate the generalization performance of the model via cross-validation:

```python
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target)
scores = cv_results["test_score"]
```

Lte's look at the mean and std:

```python
score_mean = scores.mean()
score_std = scores.std()

# print them
print(score_mean)
print(score_std)
```

*** We created a model with the default `C` value that is equal to 1. We can set this parameter:

```python
model.set_params(classifier__C=1e-3)  # here we set parameter C in the classifier
```

Let's see how model behaves with different `C`:

```python
cv_results = cross_validate(model, data, target)
scores = cv_results["test_score"]

scores.mean()
```

We got a higher score, less flexible model results in a worse prediction score. 

We can use the `get_params` method on scikit-learn models to list all the hyperparameters with their values.

```python
# loop over all parameters, and print them 
for parameter in model.get_params():
    print(parameter)
```

We can get the value of `C` as:

```python
model.get_params()["classifier__C"]
```

You see the result is `0.001`, what we set. 

Now we can automate looking for optimum value of `C`:

```python
for C in [1e-3, 1e-2, 1e-1, 1, 10]:  # loop over different values
    model.set_params(classifier__C=C)
    cv_results = cross_validate(model, data, target) 
    scores = cv_results["test_score"]
    scores_mean = scores.mean()
    print(f"Accuracy score with C={C}: {scores_mean}")
```

Looking at the results, we see with C=1 and C=10 model performs better.

### Break: 10 min

:coffee: 

### Hyperparameter tuning by grid-search

Let's check target_name:

```python
target_name
```

We sould remove the `target_name`:

```python
data = adult_census.drop(columns=[target_name, "education-num"])
```

The first step is to select all the categorical columns:
```python
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
```
Now we are going to use an estimator called `HistGradientBoostingClassifier`.
For tree-based models, the `OrdinalEncoder` avoids having high-dimensional representations:

```python
from sklearn.preprocessing import OrdinalEncoder

categorical_preprocessor = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)
```

We then use `make_column_transformer` to select the categorical columns and apply the `OrdinalEncoder` to them:

```python
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
    (categorical_preprocessor, categorical_columns),
    remainder="passthrough",
)
```

Lets now create the pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4),
        ),
    ]
)
model
```
### Tuning using a grid-search

The GridSearchCV estimator takes a param_grid parameter which defines all hyperparameters and their associated values.

```python
from sklearn.model_selection import GridSearchCV


# we tell which hyperparameter to change by defining a dictionary where
# keys are the name of hyperparameter and values are their values!
param_grid = {
    "classifier__learning_rate": (0.01, 0.1, 1, 10),  # 4 possible values
    "classifier__max_leaf_nodes": (3, 10, 30),  # 3 possible values
} 
```

Let's create a grid search:

```python
model_grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=2, cv=2)
```

First, split data to train/test:

```python
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42, test_size=0.25,
)
```

Then fit the grid search to the train data:

```python
model_grid_search.fit(data_train, target_train)
```

This might take a few seconds! Note that we define `cv=2`, cross-validation is done on the train data internally. 

We can get the results of grid search as:
```python
model_grid_search.best_params_
```

Now we get the accuracy of the model on test data using the optimum values of the hyperparameter (stores internally in `model_grid_search`):
```python
accuracy = model_grid_search.score(data_test, target_test)
accuracy
```

### Evaluation and hyperparameter tuning

We load data:

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
target_name = "class"

target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])
```

A normal cross-validation without hyperparameter tuning:

```python
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target, cv=5)

# put it in a data frame
cv_results = pd.DataFrame(cv_results)
cv_results
```

*** you need to run first the code of loading data, later the code of `categorical_columns_selector`, we have done this in the previous section. 

Split data to train/test:
```python
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)
```

```python
cv_results = cross_validate(model, data_train, target_train)
cv_results = pd.DataFrame(cv_results)
cv_results
```

Now, with hyperparameter tuning:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__learning_rate": (0.05, 0.5),
    "classifier__max_leaf_nodes": (10, 30),
}

model_grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=2, cv=2)
```

In "nested cross-validation", we use an inner cross-validation for the selection of the hyperparameters and an outer cross-validation for the evaluation of generalization performance of the refitted tuned model. 

```python
```


### Continue with the previous topic ...

After creating the grid search, we can now do nested cross validation as:

```python
# embed the grid-search in the function cross_validate
cv_results = cross_validate(
    model_grid_search, data, target, cv=5, n_jobs=2,
)
```

This might take few seconds!

Inspect the results:

```python
cv_results = pd.DataFrame(cv_results)
cv_results
```

We can check the value of the best hyperparameters obtained for each fold of the outer cross-validation: 
```python
# with return_estimator=True 
cv_results = cross_validate(
    model_grid_search, data, target, cv=5, n_jobs=2, return_estimator=True
)
```

```python
for estimator in cv_results["estimator"]:
    print(estimator.best_params_)
```

### Evaluation and hyperparameter tuning -- Exercise M3.02

The goal is to find the best set of hyperparameters which maximize the
generalization performance on a training set.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)
```

In this exercise, we progressively define the regression pipeline and later
tune its hyperparameters.

Start by defining a pipeline that:
* uses a `StandardScaler` to normalize the numerical data;
* uses a `sklearn.neighbors.KNeighborsRegressor` as a predictive model.

```python
# Write your code here.
```

Use `RandomizedSearchCV` with `n_iter=20` and
`scoring="neg_mean_absolute_error"` to tune the following hyperparameters
of the `model`:

- the parameter `n_neighbors` of the `KNeighborsRegressor` with values
  `np.logspace(0, 3, num=10).astype(np.int32)`; (import numpy as follows: `import numpy as np`)
- the parameter `with_mean` of the `StandardScaler` with possible values
  `True` or `False`;
- the parameter `with_std` of the `StandardScaler` with possible values `True`
  or `False`.

The `scoring` function is expected to return higher values for better models,
since grid/random search objects **maximize** it. Because of that, error
metrics like `mean_absolute_error` must be negated (using the `neg_` prefix)
to work correctly (remember lower errors represent better models).

Notice that in the notebook "Hyperparameter tuning by randomized-search" we
pass distributions to be sampled by the `RandomizedSearchCV`. In this case we
define a fixed grid of hyperparameters to be explored. Using a `GridSearchCV`
instead would explore all the possible combinations on the grid, which can be
costly to compute for large grids, whereas the parameter `n_iter` of the
`RandomizedSearchCV` controls the number of different random combination that
are evaluated. Notice that setting `n_iter` larger than the number of possible
combinations in a grid (in this case 10 x 2 x 2 = 40) would lead to repeating
already-explored combinations.

Once the computation has completed, print the best combination of parameters
stored in the `best_params_` attribute.

```python
# Write your code here.
```


### Solution to previous exercise M3.02

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# load data
data, target = fetch_california_housing(return_X_y=True, as_frame=True)

# scale the target
target *= 100 # this is pythonic style for doing target = target * 100
target
```

Then train/test split:

```python
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)
```

Then we define the pipeline with a regressor (because target is continuous variable):

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

scaler = StandardScaler()  # to normalize the numerical data
model = make_pipeline(scaler, KNeighborsRegressor())  # KNeighborsRegressor as a predictive model
```

Get the (pythonic) types of the values stored in the data:

```python
data.dtypes
```

For hyperparameter tuning:

```python
import numpy as np  # for working with arrays
from sklearn.model_selection import RandomizedSearchCV  # different than GridSearch

# create a dictionary for hayperparameters
param_distributions = {
    "kneighborsregressor__n_neighbors": np.logspace(0, 3, num=10).astype(np.int32), 
    "standardscaler__with_mean": [True, False],
    "standardscaler__with_std": [True, False],
}
```

*** with `astype()`, we enforce the type of data. 
*** you can check the documentation of `np.logspace` with `?` as:
```
?np.logspace
```
As can be seen, the base of the log is 10.


```python
model_random_search = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    scoring="neg_mean_absolute_error",  
    n_iter=20,
    n_jobs=2,
    verbose=1,  # print out a bit of info
    random_state=1,
)
```
*** more on `neg` in `neg_mean_absolute_error`, see https://scikit-learn.org/stable/api/sklearn.metrics.html

```python
model_random_search.fit(data_train, target_train)
```

```python
model_random_search.best_params_
```

To get the complete results:

```python
pd.DataFrame(model_random_search.cv_results_)
```

### Try out learned skills on Ames Housing dataset -- Final exercise
In this exercise we use the Ames Housing dataset.

We use this dataset in a regression setting to predict the sale prices of houses based on house features. That is, the goal is to predict the target `SalePrice` from numeric and/or categorical features in the dataset.

Remember to explore the dataset before building models. Then, start simple and step-by-step expand your approach to create better and better models.

You can load the data as follows:
```python
house_prices = pd.read_csv("../datasets/ames_housing_no_missing.csv")
```

#### In case exercise is difficult

To help you get started, you can take a look at the following link: https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_ames_housing.html


#### Scores
0.8831 (no param testing)

*** if you use scikit-learn, you can center your grid around default values. Those can be found in the documentation of an estimator, for example checkdefault value of the`learning_rate` in [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor).


### Remarks:

- Validation and evaluation matter
- Machine learning is a small part of the problem most of the times
- Technical craft is not all: Once you know how to run the software, the biggest challenges are understanding the data, its shortcomings, and what can and cannot be concluded from an analysis.
- How the predictions are used: Errors mean different things
- The predictions may modify how the system works
- Choice of the output/the labeled dataset


## Post survey 

[Survey link](https://esciencecentertraining.limesurvey.net/779524?lang=en)



## 📚 Resources

### Detailed introduction to machine learning

Introduction to Statistical Learning: https://www.statlearning.com/.


### California housing dataset

California housing dataset alternative download link: https://surfdrive.surf.nl/s/TPFeRd8AaAn9jef (expires 12/06)

Load it with
```python
import pandas as pd
data = pd.read_csv("housing_data_features.csv")
target = pd.read_csv("housing_data_targets.csv")

```

### Online MOOC on sklearn

(full version of the course material)
https://inria.github.io/scikit-learn-mooc/


### Scikit-learn API

https://scikit-learn.org/stable/index.html


### More about Decision Trees

https://scikit-learn.org/stable/modules/tree.html#decision-trees


### More about Histogram-Based Gradient Boosting (e.g. HistGradientBoostingClassifier)

https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting

