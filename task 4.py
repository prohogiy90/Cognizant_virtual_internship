import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector

import warnings
warnings.filterwarnings('ignore')


# Load data
def load_data(path: str = "/path/to/csv/"):
    """
    Function takes a path string to a CSV file and loads it into
    a Pandas DataFrame.

    :param      path: str, relative path of the CSV file

    :return     df: pd.DataFrame
    """

    data = pd.read_csv(f"{path}")
    data.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return data


# Create target variable and predictor variables
def create_target_and_predictors(
    data: pd.DataFrame = None, 
    target: str = "estimated_stock_pct"
):
    """
    Function takes in a Pandas DataFrame and splits the columns
    into a target column and a set of predictor variables, i.e. X & y.
    These two splits of the data will be used to train a supervised 
    machine learning model.

    :param      data: pd.DataFrame, dataframe containing data for the 
                      model
    :param      target: str (optional), target variable that you want to predict

    :return     X: pd.DataFrame
                y: pd.Series
    """

    # Check to see if the target variable is present in the data
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y


# Train models
def train_models_with_cv(
        X: pd.DataFrame = None,
        y: pd.Series = None
):
    """
    Function takes a predictor and target variables,
    normalizes numeric variables, transforms categorical
    variables and trains 2 models - RandomForest and Ridge.
    Based on the results of cross-validation, the average
    value of the metric and the standard deviation for each model
    are calculated and the model with the best metric is selected.

    :param      X: pd.DataFrame, predictor variables
    :param      y: pd.Series, target variable

    :return
    """

    feature_transform = ColumnTransformer(transformers=[
        ('numerical_transform', StandardScaler(), make_column_selector(dtype_include='number')),
        ('categorical_transform', OneHotEncoder(handle_unknown='ignore'), make_column_selector(
            dtype_include=object))],
        n_jobs=-1)
    X = feature_transform.fit_transform(X)

    models = (
        Ridge(alpha=1000),
        RandomForestRegressor(criterion='poisson', max_depth=2, max_features=None,
                              min_samples_split=4, n_estimators=250, n_jobs=-1)
    )

    # Evaluation of each of the models using cross-validation
    scoring = make_scorer(mean_absolute_error)

    best_score = 1E50
    best_model = None
    for model in models:
        cv_score = pd.DataFrame(cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring=scoring))
        print(f'model: {str(model).split("(")[0]}, mae_mean: {cv_score.mean()[0]:.4f}, mae_std: '
              f'{cv_score.std()[0]:.4f}')

        if cv_score.mean()[0] < best_score:
            best_score = cv_score.mean()[0]
            best_model = model

    best_model.fit(X, y)
    print(f'best model: {str(best_model).split("(")[0]}, MAE: {best_score:.4f}')


# Importing dataframe from prepared csv-file
df = load_data('C:/Users/Konstantin/Documents/Стажировка Cognizant/task 4/prepared.csv')

# Initializing target and predictor variables
X, y = create_target_and_predictors(df)

# Training and Evaluating models with cross validation
train_models_with_cv(X, y)
