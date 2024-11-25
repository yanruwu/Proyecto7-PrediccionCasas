import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_cats(data, columns, rv, plot_type="bar", plot_size=(15,12), estimador="mean", paleta="mako"):
    """
    Crea un gráfico de barras o de caja para cada columna en un conjunto de datos, agrupado por una variable de referencia.

    Esta función permite generar gráficos para comparar diferentes columnas del conjunto de datos, 
    usando una variable de referencia (`rv`). Puede elegir entre gráficos de barras (con estimadores) 
    o gráficos de caja para visualizar la distribución de los datos. Es útil para explorar y comparar 
    distribuciones categóricas en función de una variable de agrupamiento.

    Parameters:
        data (DataFrame): El conjunto de datos en formato `pandas.DataFrame` que contiene las columnas a graficar.
        columns (list): Lista de nombres de las columnas de `data` que se quieren graficar.
        rv (str): Nombre de la columna en `data` que se usará como variable de referencia (en el eje x).
        plot_type (str, optional): Tipo de gráfico a generar. Puede ser "bar" (gráfico de barras) o "box" (gráfico de caja). 
                                   El valor por defecto es "bar".
        plot_size (tuple, optional): Tamaño de la figura (ancho, alto). El valor por defecto es (15, 12).
        estimador (str, optional): Función de estimación a usar para el gráfico de barras. El valor por defecto es "mean".
        paleta (str, optional): Nombre de la paleta de colores a usar en el gráfico. El valor por defecto es "mako".

    Returns:
        None: Muestra el gráfico generado con `plt.show()`.
    """
    nrows = math.ceil(len(columns)/2)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=plot_size)
    axes = axes.flat

    for i, col in enumerate(columns):
        if plot_type.lower() == "bar":
            sns.barplot(data=data, y=col, x=rv, ax=axes[i], estimator=estimador, palette=paleta)
        elif plot_type.lower() == "box":
            sns.boxplot(data=data, y=col, x=rv, ax=axes[i], palette=paleta)

    if len(columns) % 2 != 0:
        plt.delaxes(axes[-1])
    plt.tight_layout()
    plt.show()
