import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_kaggle_env():
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE','')

def visualize_categorical_column_distribution(df, column, title, path=None):

    """
    Visualize distribution of the given categorical column on the given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given categorical column

    column: str
        Name of the categorical column

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    value_counts = df[column].value_counts()
    n = value_counts.sum()

    fig, ax = plt.subplots(figsize=(24, df[column].value_counts().shape[0] + 4), dpi=100)
    ax.bar(
        x=np.arange(len(value_counts)),
        height=value_counts.values,
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks(
        np.arange(len(value_counts)),
        [
            f'{value}\n{count:,} ({(count / n * 100):.2f}%)' for value, count in value_counts.to_dict().items()
        ]
    )
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_continuous_column_distribution(df, column, title, path=None):

    """
    Visualize distribution of the given continuous column on the given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given continuous column

    column: str
        Name of the continuous column,

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, 6), dpi=100)
    ax.hist(df[column], bins=16)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(
        title + f'''
        {column}
        Mean: {np.mean(df[column]):.2f} Std: {np.std(df[column]):.2f}
        Min: {np.min(df[column]):.2f} Max: {np.max(df[column]):.2f}
        ''',
        size=20,
        pad=12.5,
        loc='center',
        wrap=True
    )

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)
        
        
def visualize_continuous_columns_relationship(df, column1, column2, group, title, path=None):

    """
    Visualize relationship of two given continuous columns on the given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given continuous columns

    column1: str
        Name of the first continuous column
        
    column2: str
        Name of the second continuous column
        
    group: str or None
        Name of the group column

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    sns.scatterplot(x=column1, y=column2, hue=group, data=df, ax=ax)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel(column1, size=20, labelpad=15)
    ax.set_ylabel(column2, size=20, labelpad=15)
    if group is not None:
        legend = ax.legend(title=group, loc='lower right', prop={'size': 14})
        legend.get_title().set_fontsize(15)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)
