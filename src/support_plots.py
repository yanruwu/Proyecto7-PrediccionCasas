import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_cats(data, columns, rv, plot_type = "bar", plot_size = (15,12), estimador = "mean", paleta = "mako"):
    nrows = math.ceil(len(columns)/2)
    fig, axes = plt.subplots(nrows = nrows, ncols = 2, figsize = plot_size)
    axes = axes.flat

    for i, col in enumerate(columns):
        if plot_type.lower() == "bar":
            sns.barplot(data = data, y = col, x = rv, ax = axes[i], estimator=estimador, palette=paleta)
        elif plot_type.lower() == "box":
            sns.boxplot(data = data, y = col, x = rv, ax = axes[i], palette=paleta)

    if len(columns)%2 != 0:
        plt.delaxes(axes[-1])
    plt.tight_layout()
    plt.show()