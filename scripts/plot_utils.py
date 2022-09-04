import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_count(df: pd.DataFrame, column: str) -> None:
    plt.figure(figsize=(20, 10))
    fig = sns.countplot(data=df, x=column,
                        order=df[column].value_counts().index)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(column, fontsize=20)
    plt.ylabel('Absolute frequencies', fontsize=20)
    for i in range(len(df[column].value_counts())):
        fig.text(i, (df[column].value_counts().values[i]), str(df[column].value_counts().values[i]),
                 fontdict=dict(color='black', fontsize=30), horizontalalignment='center')


def multi_plot_count(df:pd.DataFrame, column:str, hue: str) -> None:
    plt.figure(figsize=(20,10))
    fig = sns.countplot(data = df, x = column, hue=hue)
    plt.title(f'Distribution of {column} per {hue}', size=20, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(column, fontsize=20)
    plt.ylabel('Absolute frequencies', fontsize=20)

    for p in fig.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        fig.annotate(f'{(height)}', (x + width/2, y + height*1.02), ha='center', size = 15)