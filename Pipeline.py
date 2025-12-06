

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRFRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score


# Function for Data Loading
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function for Data Cleaning (if needed)
def clean_data(data):
    
    data.drop('country',axis=1,inplace=True)

    data['year'].fillna(data['year'].mode()[0],inplace=True)
    data['body'].fillna(data['body'].median(),inplace=True)
    data['acidity'].fillna(data['acidity'].median(),inplace=True)
    data['type'].fillna(data['type'].mode()[0],inplace=True)
    
    data['year'].replace('N.V.',data['year'].mode()[0],inplace=True)
    data['year'] = data['year'].astype(int)

    values = data['type'].unique()
    abc = np.arange(1,22)
    for x,y in zip(values,abc):
        data['type'].replace(x,y,inplace=True)
    values1 = data['wine'].unique()
    abc1 = np.arange(1,848)

    for x,y in zip(values1,abc1):
        data['wine'].replace(x,y,inplace=True)


    # Changing string datatype to int in 'winery' column

    values2 = data['winery'].unique()
    abc2 = np.arange(1,481)

    for x,y in zip(values2,abc2):
        data['winery'].replace(x,y,inplace=True)


    # Changing string datatype to int in 'type' column

    values3 = data['region'].unique()
    abc3 = np.arange(1,77)

    for x,y in zip(values3,abc3):
        data['region'].replace(x,y,inplace=True)

    return data

# Function for Feature Selection
def select_features(X, y, k='all'):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    return X_new, selected_features

# Function for Splitting Data
def split_data(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # convert y_test to a pandas dataframe
    y_test_df = pd.DataFrame({'Price':y_test})

    # save y_test_df to a csv file
    test_csv = 'C:\\Users\\KLIN\\Documents\\Python_practice_files\\Project-Winery\\y_test.csv'
    y_test_df.to_csv(test_csv)
    
    return X_train, X_test, y_train, y_test,test_csv

# Function for Hyperparameter Tuning
def tune_hyperparameters(X_train, y_train,reg_algo):
    param_grid = {'n_estimators': [50, 100, 150],
                  'max_depth': [None, 5, 10, 15],
                  'min_samples_split': [2, 5, 10]}

    if reg_algo == "RandomForestRegressor":
        mod = RandomForestRegressor()
    elif reg_algo == "GradientBoostingRegressor":
        mod = GradientBoostingRegressor()
        
    grid_search = GridSearchCV(mod, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    #print('\nRandom Forest algorithm is used here to train the model')
    return best_model, best_params

# Function for Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)  #MSE
    RMSE = mean_squared_error(y_test, y_pred,squared=False) #RMSE
    r2 = r2_score(y_test, y_pred)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    
    # report error
    print('\nMean Squared Error: ',MSE)
    print('Root Mean Squared Error: ',RMSE)
    print('R2 score: ',r2)
    print('Mean Absolute Precentage Error :',MAPE)
    
    return MSE,RMSE,r2,MAPE,y_pred

def process(file_path):
    
    # Step 1: Load Data
    data = load_data(file_path)
    
    # Step 2: Data Cleaning
    data = clean_data(data)
    
    # Step 3: Prepare Data
    y = data['price']
    X = data.loc[:,data.columns != 'price']  # Selecting all the columns except 'price'
   
    # Step 4: Feature Selection
    X_new, selected_features = select_features(X, y)
    
    # Step 5: Split Data
    X_train, X_test, y_train, y_test,test_csv = split_data(X_new, y)
    
    reg_algo=[RandomForestRegressor,GradientBoostingRegressor]
    
    for alg in reg_algo:
        print(f'\nThe algorithm used for training is {alg.__name__} and its accuracy metrics are as follows:\n')
    
        # Step 6: Hyperparameter Tuning
        best_model, best_params = tune_hyperparameters(X_train, y_train,alg.__name__)
        best_model.set_params(**best_params)  # Set best parameters to the model
        
        # Step 7: Train Model with Best Parameters
        best_model.fit(X_train, y_train)

        # Step 8: Evaluate Model - Before Hyperparameter Tuning
        
        if alg.__name__ == "RandomForestRegressor":
            base_model = RandomForestRegressor(random_state=42)
        elif alg.__name__ == "GradientBoostingRegressor":
            base_model = GradientBoostingRegressor(random_state=42)
        
                        
        base_model.fit(X_train, y_train)
        # base_mse,base_rmse, base_r2_score, base_mape = evaluate_model(base_model, X_test, y_test)

                # Step 9: Evaluate Model - After Hyperparameter Tuning
        tuned_mse,tuned_rmse, tuned_r2_score, tuned_mape, y_pred = evaluate_model(best_model, X_test, y_test)
        
        # convert y_pred to a pandas series and save it only for RandomForestRegressor
        if alg.__name__ == "RandomForestRegressor":
            y_pred_series = pd.DataFrame({'Price': y_pred})
            pred_csv = 'C:\\Users\\KLIN\\Documents\\Python_practice_files\\Project-Winery\\wine_price_predictions.csv'
            y_pred_series.to_csv(pred_csv)
        else:
            pred_csv = 'C:\\Users\\KLIN\\Documents\\Python_practice_files\\Project-Winery\\wine_price_predictions.csv'
        

    print("Best Parameters:", best_params)
    print(f'\nThe Prediction output file is stored at {pred_csv}')
    print(f'The Test file is stored at {test_csv}')
    return tuned_mse, best_params

