# -*- coding: utf-8 -*-
import xgboost as xg
# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(____, ____, ____=____, random_state=123)

# Instantiate the XGBRegressor: xg_reg
xg_reg = ____

# Fit the regressor to the training set
____

# Predict the labels of the test set: preds
preds = ____

# Compute the rmse: rmse
rmse = ____(____(____, ____))
print("RMSE: %f" % (rmse))
