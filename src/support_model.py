# -------------------- VISUALIZACIONES --------------------
# ---------------------------------------------------------------
# Librerías utilizadas para crear gráficos, visualizar datos y resultados
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- MODELOS Y REGRESIÓN --------------------
# ---------------------------------------------------------------
# Librerías para crear y ajustar los modelos de regresión
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# -------------------- PREPROCESAMIENTO Y SELECCIÓN DE DATOS --------------------
# ---------------------------------------------------------------
# Funciones necesarias para dividir los datos en entrenamiento y prueba,
# y realizar la búsqueda de hiperparámetros con GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV

# -------------------- MÉTRICAS --------------------
# ---------------------------------------------------------------
# Funciones para calcular las métricas de rendimiento de los modelos
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

# -------------------- OTRAS --------------------
# ---------------------------------------------------------------
# Librerías adicionales que podrían usarse para procesamiento y análisis de datos
import numpy as np
import pandas as pd



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


class RegressionModel:
    def __init__(self, X, y, test_size=0.3, random_state=42):
        # División de los datos en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.model = None
        self.metrics_df = None
        self.best_params = None
        self.random_state = random_state
    
    def _get_model(self, model_type, learning_rate=0.1):
        # Diccionario de modelos disponibles
        models = {
            "linear": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(random_state=self.random_state),
            "random_forest": RandomForestRegressor(random_state=self.random_state),
            "gradient_boosting": GradientBoostingRegressor(random_state=self.random_state, learning_rate=learning_rate),
        }
        if model_type not in models:
            raise ValueError(f"El modelo '{model_type}' no es válido. Elija uno de {list(models.keys())}")
        return models[model_type]

    def train(self, model_type, params=None, learning_rate=0.1):
        # Obtener el modelo seleccionado
        self.model = self._get_model(model_type, learning_rate)
        
        # Si se pasan parámetros, se realiza GridSearch
        if params:
            grid_search = GridSearchCV(self.model, param_grid=params, cv=5, scoring="r2", n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
        
        else:
            self.model.fit(self.X_train, self.y_train)
        
        # Predicciones para las métricas
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # Crear un dataframe con las métricas
        self.metrics_df = pd.DataFrame({
            "Train": [
                r2_score(self.y_train, y_train_pred),
                mean_absolute_error(self.y_train, y_train_pred),
                root_mean_squared_error(self.y_train, y_train_pred)
            ],
            "Test": [
                r2_score(self.y_test, y_test_pred),
                mean_absolute_error(self.y_test, y_test_pred),
                root_mean_squared_error(self.y_test, y_test_pred)
            ]
        , }, index=["R2", "MAE", "RMSE"]).T
        
        return self.metrics_df

    def display_metrics(self):
        # Mostrar las métricas si están disponibles
        if self.metrics_df is not None:
            display(self.metrics_df)
        else:
            print("No hay métricas disponibles. Primero entrena el modelo.")
    
    def plot_residuals(self):
        # Verificar que se ha entrenado el modelo
        if self.model is None:
            print("Primero debes entrenar un modelo para graficar los residuos.")
            return
        
        # Predicciones para calcular los residuos
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)


        # Crear gráficos de los residuos
        plt.figure(figsize=(12, 6))

        # Residuos de entrenamiento
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=self.y_train, y=y_train_pred, color="blue", alpha=0.6)
        plt.plot([min(self.y_train),max(self.y_train)], [min(y_train_pred), max(y_train_pred)], color = "red", ls = "--")
        plt.title("Train")
        plt.xlabel("Valores Reales")
        plt.ylabel("Valores predichos")

        # Residuos de prueba
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=self.y_test, y=y_test_pred, color="orange", alpha=0.6)
        plt.plot([min(self.y_test),max(self.y_test)], [min(y_test_pred), max(y_test_pred)], color = "red", ls = "--")
        plt.title("Test")
        plt.xlabel("Valores Reales")
        plt.ylabel("Valores predichos")

        plt.tight_layout()
        plt.show()
    
    def get_best_params(self):
        # Obtener los mejores parámetros si se realizaron búsquedas en cuadrícula
        if self.best_params:
            return self.best_params
        else:
            print("No se ha realizado búsqueda en cuadrícula o no hay parámetros disponibles.")
            return None
