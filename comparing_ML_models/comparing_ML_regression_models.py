# Assignment from DSCI 571 Lab 4
# Predict abalone age using 8 features (sex, length, diameter, height, whole weight, shucked weight, viscera weight, shell weight) and 1 response (rings)

# Import libraries
import numpy as np
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# 1. Load data
abalone_df = pd.read_csv('abalone_age.csv')

# 2. Check data for outliers/erroneous values
abalone_df.describe()
# does not look like there are any outliers or erronuous values (minimum and maximum values are not extremely far off from the mean; standard deviations make sense)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(abalone_df.drop(columns = 'Rings'), abalone_df[['Rings']], test_size=0.2)

# 4. Preprocess data
numeric_features = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']
categorical_features = ['Sex']

preprocessor = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), numeric_features),
        ('ohe', OneHotEncoder(drop="first"), categorical_features)])

X_train = pd.DataFrame(preprocessor.fit_transform(X_train),
                       index=X_train.index,
                       columns=(numeric_features +
                                list(preprocessor.named_transformers_['ohe']
                                     .get_feature_names(categorical_features))))
X_test = pd.DataFrame(preprocessor.transform(X_test),
                      index=X_test.index,
                      columns=X_train.columns)
                      
# 5. Fit models
lr = LinearRegression()
lr.fit(X_train, y_train)

svr = SVR(gamma = 'scale')
parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 10]}
svr_cv = GridSearchCV(svr, parameters, cv = 5)
svr_cv.fit(X_train, y_train.to_numpy().ravel())

knr = KNeighborsRegressor()
parameters = {'n_neighbors':[1, 50]}
knr_cv = GridSearchCV(knr, parameters, cv = 5)
knr_cv.fit(X_train, y_train)

rfr = RandomForestRegressor()
parameters = {'n_estimators':[1, 200], 'max_depth':[1, 50]}
rfr_cv = GridSearchCV(rfr, parameters, cv = 5)
rfr_cv.fit(X_train, y_train.to_numpy().ravel())

# 6. Report test error
print("Linear Regression Root Mean Squared Error: %.4f" % np.sqrt(mean_squared_error(y_test, lr.predict(X_test))))
print("SVR Root Mean Squared Error: %.4f" % np.sqrt(mean_squared_error(y_test, svr_cv.predict(X_test))))
print("k Neighbours Regressor Root Mean Squared Error: %.4f" % np.sqrt(mean_squared_error(y_test, knr_cv.predict(X_test))))
print("Random Forest Regressor Root Mean Squared Error: %.4f" % np.sqrt(mean_squared_error(y_test, rfr_cv.predict(X_test))))

# 7. Produce plot to compare models
data = []
for i in range(len(X_test)):
    data.append(['Linear Regression', lr.predict(X_test)[i][0] + 1.5, y_test.to_numpy().ravel()[i] + 1.5])
    data.append(['SVR', svr_cv.predict(X_test)[i] + 1.5, y_test.to_numpy().ravel()[i] + 1.5])
    data.append(['k Neighbours Regressor', knr_cv.predict(X_test)[i][0] + 1.5, y_test.to_numpy().ravel()[i] + 1.5])
    data.append(['Random Forest Regressor', rfr_cv.predict(X_test)[i] + 1.5, y_test.to_numpy().ravel()[i] + 1.5])
df = pd.DataFrame(data, columns=['Model', 'Predicted Age', 'Observed Age'])

df_line = pd.DataFrame([[0,0], [25,25]], columns=['x','y'])

scatter = alt.Chart(df, title = 'Predicted Age vs. Observed Age').mark_circle(opacity = 0.3).encode(
    x = alt.X('Observed Age:Q'),
    y = alt.Y('Predicted Age:Q'),
    color = alt.Color('Model:N')
    ).properties(width = 1000, height = 500)

line = alt.Chart(df_line).mark_line().encode(
    x = alt.X('x:Q'),
    y = alt.Y('y:Q'),
    color = alt.value('red')
    ).properties(width = 1000, height = 500)

line + scatter

# SVR does best as it has the lowest root mean squared error. However, the models are very similar in their performances.
