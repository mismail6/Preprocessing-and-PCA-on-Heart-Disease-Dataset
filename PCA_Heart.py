import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load heart disease dataset in pandas dataframe
# Remove outliers using Z score. Usual guideline is to remove anything that has Z score > 3 formula or Z score < -3
# Convert text columns to numbers using label encoding and one hot encoding
# Apply scaling
# Build a classification model using various methods (SVM, logistic regression, random forest) and check which model
# gives you the best accuracy
# Now use PCA to reduce dimensions, retrain your model and see what impact it has on your model in terms of accuracy.
# Keep in mind that many times doing PCA reduces the accuracy but computation is much lighter and that's the trade off
# you need to consider while building models in real life

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('heart.csv')

relevant_columns = ['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
d1 = df[relevant_columns]

# Calculate Z-scores for each column
z_scores = d1.apply(zscore)
no_outliers = d1[(z_scores.abs() <= 3).all(axis=1)] # can use > 3 and .any to see outliers
updated_df = df.loc[no_outliers.index]

# Ordinal encoding
label_encoder = LabelEncoder()
updated_df["ST_Slope_LabelEncoded"] = label_encoder.fit_transform(updated_df["ST_Slope"])
updated_df.drop(columns=["ST_Slope"], inplace=True)
updated_df["RestingECG_Encoded"] = label_encoder.fit_transform(updated_df["RestingECG"])
updated_df.drop(columns=["RestingECG"], inplace=True)

# OneHotEncoding
encoder = OneHotEncoder(drop='first', sparse_output=False)
columns_to_encode = ['Sex', 'ChestPainType', 'ExerciseAngina']
encoded_data = encoder.fit_transform(updated_df[columns_to_encode])
encoded_column_names = encoder.get_feature_names_out(columns_to_encode)
encoded_data_df = pd.DataFrame(encoded_data, columns=encoded_column_names)
updated_df.reset_index(drop=True, inplace=True)
encoded_data_df.reset_index(drop=True, inplace=True)
data_encoded = pd.concat([updated_df.drop(columns=columns_to_encode), encoded_data_df], axis=1)

# Dropping target column
y = data_encoded['HeartDisease']
data_encoded.drop('HeartDisease', axis=1, inplace=True)

# testing dataannotation
numeric_columns = data_encoded.select_dtypes(include=['number']).columns
data_numeric = data_encoded[numeric_columns]

# print(data_encoded.columns)
# print(data_numeric.columns)

# Our processed data is READY. Now we need to scale it
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data_encoded)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_numeric)
data_scaled_df = pd.DataFrame(data_scaled, columns=numeric_columns)
print(data_scaled_df)


pca = PCA(n_components=2)
x_pca = pca.fit_transform(data_scaled)
print(pca.explained_variance_ratio_)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=30)
model = RandomForestClassifier()
model.fit(X_train_pca, y_train)
score = model.score(X_test_pca, y_test)
print(score)