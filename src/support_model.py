
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def create_model(params, X_train, y_train, method = DecisionTreeRegressor(), cv= 5, scoring = "neg_mean_squared_error"):
    grid_search = GridSearchCV(estimator = method, param_grid=params, cv = cv, scoring = scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search

def metricas(y_train, y_train_pred, y_test, y_test_pred):
    """
    Calcula métricas de regresión para conjuntos de entrenamiento y prueba.
    
    Parameters:
        y_train (array-like): Valores reales del conjunto de entrenamiento.
        y_train_pred (array-like): Predicciones del conjunto de entrenamiento.
        y_test (array-like): Valores reales del conjunto de prueba.
        y_test_pred (array-like): Predicciones del conjunto de prueba.
    
    Returns:
        dict: Diccionario con métricas de R², MAE, MSE y RMSE.
    """
    metricas = {
        'train': {
            'r2_score': r2_score(y_train, y_train_pred),
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'MSE': mean_squared_error(y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred))
        },
        'test': {
            'r2_score': r2_score(y_test, y_test_pred),
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'MSE': mean_squared_error(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred))
        }
    }
    return metricas