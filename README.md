The hreat.csv file contains several columns that influence a target column 'HeartDisease' containing binary values.

The objective is to reduce the dimensions to the two most relevant rows using PCA.
But first, preprocessing steps are necessary to achieve the desired result.

First we remove outliers by applying zscore and filtering the pandas data frame.
Then we transform the nominal columns through label encoding and one hot encoding techniques into numerical ones.
Afterwards, we scale all of the columns(now numeric) using either Standard Scaler or MinMax Scaler.
At this point our processed dataframe should be ready for any type of machine learning model.
