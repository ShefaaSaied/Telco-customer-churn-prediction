# Telco-customer-churn-prediction
A classification machine learning problem for predicting customers churn from the company based on customers who left within the last month labeled by 'yes' or 'no'

The dataset used in this project is obtained from [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)\
The data set includes information about:
- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

## Methodology
At first 20% of the data were splitted for final testing; stratified by the 'Churn' (target) column.

## Data cleaning
* Convert 'TotalCharges' column which is of object type to float type using pd.to_numeric() with errors parameter set to 'coerce' to parse invalid data to NaN.
* Eight missing values were found in the 'TotalCharges' column and were imputed by the mean() value.
* Data has no duplicates.

## Exploratory data analysis
1. Count plot shows the distribution of the churn rate in the data which showed an imbalance in the data.
2. Categorical features count plot insights:
    * Data is evenly distributed between the two genders; males and females, which might be useful in further analysis.
    * No information added by 'No Internet Service' or 'No Phone Service' and 'No' categories.
    --> **Replacing 'No Internet Service' and 'No Phone Service' entries with 'No'**.
3. Histogram and box plot of continous features implies that:
    * No outliers exists.
    * 'TotalCharges' feature is right skewed.
4. Scatter plot of 'MonthlyCharges' vs. 'TotalCharges' shows a positive correlation between both and also it affects the Churn rate positively.

## Feature encoding 
Several encoding techniques were tested on each categorical feature separately and One-Hot encoding all the categorical features gave the best results.

## Feature engineering
Binning 'tenure' feature into 6 ranges:
* 0-12 months --> '0-1 years'
* 12-24 months --> '1-2 years'
* 24-36 months --> '2-3 years'
* 36-48 months --> '3-4 years'
* 48-60 months --> '4-5 years'
* More than 60 months --> 'more than 5 years'

## Feature scaling
log transformation is very powerful in feature scaling specially with skewed data, hence, np.log1p() is applied on 'MonthlyCharges' and 'TotalCharges' features and with trials it proved giving the best results over MinMaxScaler() and StandaredScaler().

## Data imbalance
Data imbalance affects machine learning models by tending only to predict the majority class and ignoting the minority class, hence, having major misclassification of the minority class in comparison with the majority class. Hence, we use techniques to balance class distribution in the data.

Even that our data here doesn't have severe class imbalance, but handling it shows results improvement.
Using SMOTE (Synthetic Minority Oversampling Technique) libraray in python that randomly increasing the minority class which is 'yes' in our case.

SMOTE synthetically creates new records of the minority class by randomly selecting one or more of the k-nearest neighbors for each example in the minority class. Here, k= 5 neighbors is used. 

### Preparing a python function test_prep(dataframe) to combine and apply all previous preprocessing steps on the test data.
- To handle any expected missing values in the test set, a condition is added inside the function to map the mean value of its column in the train set.

## Models training
Four different models were applied on the data and all results are reported with confusion matrix and classification report showing the precision, recall, and f1-score metrics.
1. Logistic regression
Best parameters after several trials: C=200 (very large c value trying to fit the data as possible without overfitting), max_iter=1000
2. Support vector classifier
Best prameters: kernel='linear', C=20
3. XGBoost classifier
RandomizedSearchCV is used for hyperparameters tuning with StratifiedKFold of 5 splits.
4. Multi-layer Perceptron (MLP) classifier.