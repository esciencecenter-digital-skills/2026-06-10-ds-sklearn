![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 1 -- Introduction to Machine Learning (Day 3 ODISSEI summer school)

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.


----------------------------------------------------------------------------

This is the Document for today: https://edu.nl/ehx87

Collaborative Document day 1: https://edu.nl/ehx87

Collaborative Document day 2:


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
https://esciencecenter-digital-skills.github.io/2026-06-10-ds-sklearn/

### 🛠 Setup
https://esciencecenter-digital-skills.github.io/2026-06-10-ds-sklearn#setup

## 👩‍🏫👩‍💻🎓 Instructors

Malte Lüken, Flavio Hafner

## 🧑‍🙋 Helpers

Angel Daza

## 🗓️ Agenda
<div class="row">
  <div class="col-md-12">
    <table class="table table-striped">
      <tr> <th>Time</th> <th>Content</th></tr>
      <tr> <td>09:00</td>  <td>Welcome and icebreaker</td> </tr>
      <tr> <td>09:15</td>  <td>Introduction to machine learning</td> </tr>
      <tr> <td>10:45</td>  <td>Tabular data exploration (without exercise)</td> </tr>
      <tr> <td>10:10</td>  <td>Break  </td> </tr>
      <tr> <td>10:25</td>  <td>Fitting a scikit-learn model on numerical data </td> </tr>
      <tr> <td>11:00</td>  <td>Break</td> </tr>
      <tr> <td>11:10</td>  <td>Fitting a scikit-learn model on numerical data (continued) </td> </tr>
      <tr> <td>11:50</td>  <td>Preprocessing numerical features </td> </tr>
      <tr> <td>12:10</td>  <td>Lunch Break</td> </tr>
      <tr> <td>13:10</td>  <td>Preprocessing numerical features (continued) </td> </tr>
      <tr> <td>14:10</td>  <td>Break</td> </tr>
      <tr> <td>14:25 (instead of 13:30)</td>  <td>Preprocessing categorical features </td> </tr>
      <tr> <td>15:05</td>  <td>Break</td> </tr>
      <tr> <td>15:15 (instead of 14:20)</td>  <td>Combining numerical and categorical features </td> </tr>
      <tr> <td>15:35 (instead of 14:40)</td>  <td>Overfitting and underfitting </td> </tr>
      <tr> <td>15:50</td>  <td>Wrap-up</td> </tr>
      <tr> <td>16:00</td>  <td>Guest lecture</td> </tr>
      <tr> <td>17:00</td>  <td>END</td> </tr>
      <tr> <td>14:40</td>  <td>Overfitting and underfitting </td> </tr>
      <tr> <td>14:55</td>  <td>Cross-validation </td> </tr>
      <tr> <td>15:10</td>  <td>Break</td> </tr>
      <tr> <td>15:20</td>  <td>Cross-validation (continued) </td> </tr>
      <tr> <td>15:35</td>  <td>Bias-vs-variance trade-off </td> </tr>
      <tr> <td>15:55</td>  <td>Wrap-up</td> </tr>
      <tr> <td>16:00</td>  <td>Guest lecture</td> </tr>
      <tr> <td>17:00</td>  <td>END</td> </tr>
    </table>
  </div>
</div>


## 🔧 Exercises

### Machine learning concepts -- Quiz Intro.01 (Malte)
Given a case study: pricing apartments based on a real estate website. We have thousands of house descriptions with their price. Typically, an example of a house description is the following:

“Great for entertaining: spacious, updated 2 bedroom, 1 bathroom apartment in Lakeview, 97630. The house will be available from May 1st. Close to nightlife with private backyard. Price ~$1,000,000.”

We are interested in predicting house prices from their description. One potential use case for this would be, as a buyer, to find houses that are cheap compared to their market value.

#### What kind of problem is it?

*a) a supervised problem*
b) an unsupervised problem
c) a classification problem
*d) a regression problem*
 
 a supervised problem
 d regression problem

#### What are the features?

*a) the number of rooms might be a feature*
b) the post code of the house might be a feature
c) the price of the house might be a feature

a and b
Select all answers that apply

#### What is the target variable?

a) the full text description is the target
*b) the price of the house is the target*
c) only house description with no price mentioned are the target

the price of the house is the target
Select a single answer

#### What is a sample?

a) each house description is a sample
b) each house price is a sample
*c) each kind of description (as the house size) is a sample*

each kind of description is a sample 
Select a single answer


### First model with scikit-learn 

The goal of this exercise is to fit a similar model as in the previous notebook to get familiar with manipulating scikit-learn objects and in particular the `.fit/.predict/.score API`.

In the previous notebook we used model = `KNeighborsClassifier()`. All scikit-learn models can be created without arguments. This is convenient because it means that you don’t need to understand the full details of a model before starting to use it.

One of the `KNeighborsClassifier` parameters is `n_neighbors`. It controls the number of neighbors we are going to use to make a prediction for a new data point.

What is the default value of the `n_neighbors` parameter?

Hint: Look at the documentation on the scikit-learn website or directly access the description inside your notebook by running the following cell. This opens a pager pointing to the documentation.

```python
from sklearn.neighbors import KNeighborsClassifier

KNeighborsClassifier?
```

Create a KNeighborsClassifier model with `n_neighbors=50`

```python=
model_50 = KNeighborsClassifier(n_neighbors=50) # model is not trained yet
```

Fit this model on the training data and target that we used before

```python=
_ = model_50.fit(data, target)
```

Use your model to make predictions on the first 10 data points inside the data. Do they match the actual target values?

```python=
first_data_features = data.iloc[:10] # subsets the data to the first 10 rows
first_predictions = model_50.predict(first_data_features)
first_predictions
```

Compute the accuracy on the training data.

```python=
first_target_values = target.iloc[:10]
(first_predictions == first_target_values).mean()
```

```python
model_50.score(data, target)
```

Now load the test data from `"../datasets/adult-census-numeric-test.csv"` and compute the accuracy on the test data.

```python=
model_50.score(data_test, target_test)
```

### Processing numerical features -- Exercise M1.03

The goal of this exercise is to compare the performance of our classifier in the previous notebook (roughly 81% accuracy with `LogisticRegression`) to some simple baseline classifiers. The simplest baseline classifier is one that always predicts the same class, irrespective of the input data.
- What would be the score of a model that always predicts ' >50K'?
- What would be the score of a model that always predicts ' <=50K'?
- Is 81% or 82% accuracy a good score for this problem?

Use a `DummyClassifier` and do a train-test split to evaluate its accuracy on the test set. This [link](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators) shows a few examples of how to evaluate the generalization performance of these baseline models.

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
```
We first split our dataset to have the target separated from the data used to train our predictive model.

```python
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)
```


We start by selecting only the numerical columns as seen in the previous notebook.


```python
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]
```
Split the data and target into a train and test set.

```python
from sklearn.model_selection import train_test_split

# Write your code here.
```

Use a `DummyClassifier` such that the resulting classifier always predict the class ' >50K'. What is the accuracy score on the test set? Repeat the experiment by always predicting the class ' <=50K'.

Hint: you can set the strategy parameter of the DummyClassifier to achieve the desired behavior.

```python
from sklearn.dummy import DummyClassifier

# Write your code here.
```

#### Solution

Always pick the high revenue:
```python
from sklearn.dummy import DummyClassifier
class_to_predict = " >50K"
high_revenue_classifier = DummyClassifier(strategy="constant", constant=class_to_predict)
high_revenue_classifier.fit(data_train, target_train)
score_hrc = high_revenue_classifier.score(data_test, target_test)
score_hrc # ~ 0.23
```

Always pick the low revenue:
```python
class_to_predict = " <=50K"
low_revenue_classifier = DummyClassifier(strategy="constant", constant=class_to_predict)
low_revenue_classifier.fit(data_train, target_train)
score_lrc = low_revenue_classifier.score(data_test, target_test)
score_lrc # ~ 0.77
```

This actually shows the imbalance in the data classes (there are many more low revenue samples than high revenue, this is why our "dummy low revenue" classifier performs "quite ok", it just gets lucky more often because of the data distribution).

We can also immediately pick the most frequent class in the dataset with 

```python
revenue_classifier = DummyClassifier(strategy="most_frequent")
revenue_classifier.fit(data_train, target_train)
score = revenue_classifier.score(data_test, target_test)
score # ~ 0.77
```

This is also a good baseline to see how much is our ML model improving from simple assumptions.

## 🧠 Collaborative Notes

### Introduction

- Machine Learning (ML), the craft of creating predictive models. Engineering rules to predict outcomes from the data.

- In ML we care about generalizing to new observations. There is a difference between memorizing and generalizing: we have train data, our "memory"; and we have test data, "unknown data", for which our trained model will make predictions.

- Noise is unexplainable variance.

- In ML we often have tabular data. Rows are samples, columns are features.

- **Supervised learning:** Matrix *X* with *n* observations and target *y*, the label of each row. The goal is to predict *y*.
    - **Classification:** *y* is discrete.

    - **Regression:** *y* is continuous.

- **Unsupervised learning:** we don't have target *y*, the aim is to discover the structure inside matrix *X*.

- **Ensemble learning:** different models make predictions and then we can take the majority vote to make a final prediction

### Tabular Data Exploration

- Inspect your dataset before any ML model, to find peculiarities in your data and decide if ML is useful for this problem.
    - Load your data
    - Identify your target variable (what unique values does it have?). Is it **numerical**? Is it **categorical**? How many classes?
    - Look at the dataset shape. How many samples (rows)? How many features (columns)? The number of features normally is number of columns - 1.

- Inspect numerical columns: we can visualize the distribution with a histogram with `df.hist()`.
- Inspect categorical columns with `value_counts()` to see the distribution of categories, this can be called on a pandas Series (column).
- Training models on imbalanced data can cause disproportionate prediction errors for under-represented groups.
- Inspect relations between columns with `pd.crosstab()`. With this you can discover, for example, if two columns are "duplicated" information, this can harm your model so you would like to keep only one.
- Use *pairplots* to visualize numerical columns to see how each of the features and the target column are related. You can use `seaborn` to make these plots.


### First model with scikit-learn

```python
import pandas as pd 
adult_census = pd.read_csv("../datasets/adult-census-numeric.csv")
adult_census
```


```python
target_name = "class"
target = adult_census[target_name]
target
```

- We get a `pd.Series`, which is a single column in a `pd.DataFrame`
- It is crucial to separate the target from the X matrix. So far, we only referenced it. Now we'll separate it.


```python
data = adult_census.drop(columns=[target_name]) 
data # same as before, but without the class column. We'll use this dataframe for the predictor variables later

```

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier() # this instantiates the KNeighborsClassifier class
model # shows the parameters and current values of the instance
```

- In sklearn, all models are implemented in classes. They have the same methods, as we will see

The first method we use is the `fit` method, with which we can train
the model given features and targets:

```python
_ = model.fit(data, target)

```

Schema of what `model.fit` does

![](https://codimd.carpentries.org/uploads/upload_4c144073c7bb3555675ffddd957baa1d.png)

The model state stores the results from the fiting procedure. 

The second method we use is the `predict` method. It takes one input -- the features of the 
data we want to make predictions.

Here, we predict on the training set.

```python
target_predicted = model.predict(data)
target_predicted # we see that the model predicts the labels
```

Schema for the predict method:
![](https://codimd.carpentries.org/uploads/upload_8d16e4c6633ed8e338a84483a50b1d7f.png)

The model state is used to make predictions.

**What does `KNeighborsClassifier` do?**
- memorizes the training dataset
- makes a prediction for a new sample based on its `K` closest neighbors in the training dataset
- it is a very simple model, but can be a useful baseline to start and compare more complex models to
- it works best with few features; as the number of features grows, it becomes more and more difficult to find the appropriate neighbors


We can compare the predictions against the actual values of the target
```python
target == target_predicted # gives the comparison for all rows
# aggregate as follows
(target == target_predicted).mean() # takes the mean on the boolean array `(target == target_predicted)`
```

The mean correct predictions above is also called accuracy.

#### Train-test split

We now load a prepared test data set

```python
adult_census_test = pd.read_csv("datasets/adult-census-numeric-test.csv")
target_test = adult_census_test[target_name]
data_test = adult_census_test.drop(columns=[target_name])
data_test
```


```python
accuracy = model.score(data_test, target_test)
accuracy
```

What does `score` do?
- takes the train model and makes a prediction on `data_set`
- compares the prediction with `target_test`
- each model has a `score` method, but different models have a different default scoring method
- here, the `score` returns the accuracy of the classification
- we see that the model does slightly worse on the test data than on the train data

TODO: how to find the default scoring method?

```python
?model.score # gives us the help on the `score` method

```

### Preprocessing numerical features

```python
import pandas as pd
adult_census = pd.read_csv('datasets/adult-census.csv')
```

```python
adult_census = adult_census.drop(columns="education-num")
```

```python
data = adult_census.drop(columns="class")
target = adult_census["class"]
data.columns
```


```python
data[["age", "hours-per-week", "sex", "race"]].head() # see the first rows of the data
```

```python
data[["age", "hours-per-week", "sex", "race"]].dtypes # see the type of each column
```

```python
data.dtypes # for the full dataset
```


```python
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
data_numeric = data[numerical_columns]
data[numerical_columns].head()
```

```python
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, 
    target, 
    random_state=42, 
    test_size=0.25
)
```



```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```


```python
_ = model.fit(data_train, target_train)
```


```python
accuracy = model.score(data_test, target_test)
accuracy
```

```python
data_train.describe() # summary stats per column (mean, stf, etc...)
```

The scaler normalizes the columns ranges based on the mean and standard deviation of each column, this improves algorithm performance
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data_train)
```

```python
scaler.mean_
```

```python
scaler.scale_
```

We scale the data based on the mean and std information
```python
data_train_scaled = scaler.transform(data_train)
data_train_scaled
```

Now each column of the data will have a standard deviation of 1 and a mean of 0.

```python
data_train_scaled[:,0].std()
```

```python
data_train_scaled[:,0].mean()
```
### Using a pipeline

We can chain different steps in a single model so we make sure we always execute them in order. This is called a pipeline

![](https://codimd.carpentries.org/uploads/upload_f61c5390d1d4fc5f612fdf5caeb0d0ce.png)


```python
from sklearn.pipeline import make_pipeline
model = make_pipeline(StandardScaler(), LogisticRegression())
```

```python
model.fit(data_train, target_train)
```

```python
predicted_target = model.predict(data_test)
```

```python
predicted_target[:5]
```

Get the accuracy scores for the model pipeline:
```python
score = model.score(data_test, target_test)
score
```

### Encoding Strategies

```python
import pandas as pd

adult_census = pd.read_csv("datasets/adult-census.csv")
adult_census = adult_census.drop(columns="education-num")

target_name = "class"
target = adult_census[target_name]

data = adult_census.drop(columns=[target_name])
```

```python
data["native-country"].value_counts().sort_index()
```

```python
# These are the type of each column in the data
data.dtypes
```

```python
from sklearn.compose import make_column_selector as selector
# only select columns of a certain types for example `object` (for OLDER versions)
categorical_column_selector = selector(dtype_include=object)
# only select columns of a certain types for example `string` (for NEWER versions)
categorical_column_selector = selector(dtype_include=str)
```

```python
categorical_columns = categorical_column_selector(data)
categorical_columns
```

Only keep the categorical columns for all samples:
```python
data_categorical = data[categorical_columns]
data_categorical
```

#### Ordinal Encoder

**Ordinal encoding** assigns an integer to each category. It indicates an ORDER within the values, so it is useful when the order is meaningful.
```python
from sklearn.preprocessing import OrdinalEncoder
education_column = data_categorical[["education"]]
education_column
```

```python
encoder = OrdinalEncoder().set_output(transform="pandas")
education_encoded = encoder.fit_transform(education_column)
education_encoded
```

To show the order of the labels that the encoder used to infer the numerical values: 
```python
encoder.categories_
```
The default is alphabetical order (which in this case is not correct). You can change the order:

```python
OrdinalEncoder(categories=["define", "here", "the ordered categories..."])
```

#### One-Hot Encoder

If we do not care about the order of the assignments (it is not meaningful) is better to use the **OneHotEncoder**, to avoid false assumptions about the ordering
```python
from sklearn.preprocessing import OneHotEncoder
# we use this `sparse_output=False` arg for making visualization more intuitive (include all the zeroes), but it is more efficient to leave it in the default value for big datasets
encoder = OneHotEncoder(sparse_output=False)
education_encoded = encoder.fit_transform(education_column)
education_encoded
```

We can apply the one hot encoder to the entire dataset:
```python
encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
data_encoded = encoder.fit_transform(data_categorical)
data_encoded
```

### Combining numerical and categorical features

Select features based on data dtype with selectors:
```python
from sklearn.compose import make_column_selector as selector

# Exclude all columns with object dtype
numerical_columns_selector = selector(dtype_exclude=object)
# Include all columns with object dtype
categorical_columns_selector = selector(dtype_include=object)
```

```python
# Select numerical columns with selector
numerical_columns = numerical_columns_selector(data)
# Select categorical columns with selector
categorical_columns = categorical_columns_selector(data)
```

Dispatch different preprocessors for different column types:

```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

```

```python
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
    (categorical_preprocessor, categorical_columns),
    (numerical_preprocessor, numerical_columns),
)

```

The preprocessor applies the `OneHotEncoder` to the categorical columns and the `StandardScaler` to the numerical columns:

![](https://codimd.carpentries.org/uploads/upload_eb862c5d79edb4af001e10adffaaea33.png)

We can use the preprocessor in a pipeline together with a model:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=500),  # Only use max 500 iterations when fitting the log reg
)
```

Evaluate the pipeline using a train-test split:

```python
from sklearn.model_selection import train_test_split

# Split into train and test set
data_train, data_test, target_train, target_test = train_test_split(
    data,
    target,
    random_state=42,
)

# Train the pipeline
model.fit(data_train, target_train)

# Evaluate the trained pipeline on the test set
score = model.score(data_test, target_test)
# ~0.85
```

Using both numerical and categorical columns improves the prediction performance on the test set.


## Overfitting and underfitting

Understand when and why a model does or does not generalize well on data it has not seen.
- underfitting: model is not flexible enough for the data
- overfitting: model is too flexible (complex)
- Example of model complexity: 9th-degree polynomial -- it is a very flexible curve
   - higher degree polynomial = more flexible
   - we can vary the model complexity with the polynomial degree
- Gonig from less to more complex
   - degree 1 polynomial -- linear: does not do well
   - degree 5 polynomial does quite well
   - degree 9 polynomial seems to be overfitting, even though the true data generating process is 9th degree polynomial
   - reason for this: noise in the data -> noise can imply that a simpler model is better than a complex model
- overfitting: model is too complex. Its flexibility captures noise, even though it approximate well the data-generating process
- underfitting: model is too simple for the data -- can happen when we have a large dataset and small amount of noise
- *How to find the right trade-off between overfitting and underfitting*


## 📚 Resources

- Course material: https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/

