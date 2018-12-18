import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import torch
from torch.distributions import Categorical, Beta, Dirichlet, Uniform
from torch.distributions import MultivariateNormal as Normal
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from pylab import rcParams

sns.set()
rcParams['figure.figsize'] = 10,7

def parameter(*args):
    return torch.from_numpy(np.array(args, dtype=np.float32))

def plot(data, true_means=None, z=None):
    # Prep data
    data = pd.DataFrame(data)
    data = data.rename(index=str, columns={0: 'x', 1: 'y', 2: 'comp'})
    data['type'] = 'data'

    if true_means is not None:
        # Add clusters
        clusters = [(datum.numpy()[0], datum.numpy()[1], component) for component, datum in enumerate(true_means)]
        clusters = pd.DataFrame(clusters)
        clusters = clusters.rename(index=str, columns={0: 'x', 1: 'y', 2: 'comp'})
        clusters['type'] = 'truth'

        df = pd.concat([data, clusters])
        ax = sns.scatterplot(x="x", y="y", hue='comp',  size="type", style="type",
                             sizes=(150, 25), legend=False, data=df)

    else:
        df = data
        ax = sns.scatterplot( x="x", y="y", data=df)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(0, 15)
    ax.set_xlim(0, 15)

    if z is not None:
        ax.set_title("Generating from: Cluster {}".format(z))
    else:
        ax.set_title("Observations")


def plot_stick(measure, arrow=None, active=[], labels=None):
    df = pd.DataFrame(measure.numpy()).T
    ax = df.plot(kind='barh', stacked=True, ec='black', legend=False, xticks=[0, 1], figsize=(12,2))
    x_axis = ax.axes.get_yaxis()
    x_axis.set_visible(False)
    ax.set_facecolor('w')
    if len(measure) < 20:
        for idx, p in enumerate(ax.patches):
            if labels is not None:
                ax.annotate(labels[idx], (p.get_x() * 1.005, p.get_height() / 2.9))
            else:
                ax.annotate(str('p_{}'.format(idx)), (p.get_x() * 1.005, p.get_height() / 2.9))
    if arrow is not None:
        ax.arrow(arrow, -0.37, 0, 0.05, head_width=0.05, head_length=0.1, fc='k', ec='k')
    plt.autoscale()


def kmeans(data):
    x,y,_ = zip(*data)
    X = np.array(list(zip(x,y)))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    mu0, mu1 = kmeans.cluster_centers_
    p1 = kmeans.labels_.sum() / len(kmeans.labels_)
    p0 = 1 - p1
    return mu0, mu1, p0, p1


def plot_stick_and_data(data, components, component_to_mean_map, arrow=None):
    # Plot figure with subplots of different sizes
    fig = plt.figure(1)
    # set up subplot grid
    gridspec.GridSpec(4,1)

    # Set color map
    grey = sns.xkcd_palette(["greyish"])
    n_components = len(components) if (components.dim() != 0) else 1

    huemap = {i:grey for i in range(n_components)}
    palette = sns.color_palette(n_colors=len(component_to_mean_map))
    for idx, comp in enumerate(component_to_mean_map.keys()):
        huemap[comp] = palette[idx]

    # Prep data
    data = pd.DataFrame(data)
    data = data.rename(index=str, columns={0: 'x', 1: 'y', 2: 'comp'})
    data['type'] = 'data'

    # Add clusters
    clusters = [(datum.numpy()[0], datum.numpy()[1], component) for component, datum in component_to_mean_map.items()]
    clusters = pd.DataFrame(clusters)
    clusters = clusters.rename(index=str, columns={0: 'x', 1: 'y', 2: 'comp'})
    clusters['type'] = 'truth'
    df = pd.concat([data, clusters])

    # large subplot
    ax1 = plt.subplot2grid((4,1), (0,0), colspan=1, rowspan=3)

    ax = sns.scatterplot(x="x", y="y",
                    hue='comp',
                    size="type",
                    style="type",
                    sizes=(150, 25),
                    palette=huemap,
                    legend=False,
                    data=df)

    ax.set_ylim(0, 15)
    ax.set_xlim(0, 15)

    # Add stick
    ax2 = plt.subplot2grid((4,1), (3,0), colspan=1, rowspan=1)
    df = pd.DataFrame(components.numpy()).T

    ax = df.plot(kind='barh',
                 stacked=True,
                 ec='black',
                 legend=False,
                 color=[huemap[i] for i in range(len(huemap))],
                 ax=ax2)

    x_axis = ax.axes.get_yaxis()
    x_axis.set_visible(False)
    ax.set_facecolor('w')
    if len(components) < 20:
        for idx, p in enumerate(ax.patches):
            ax.annotate(str('p_{}'.format(idx)), (p.get_x() * 1.005, p.get_height() / 2.9))
    if arrow is not None:
        ax.arrow(arrow, -0.37, 0, 0.05, head_width=0.05, head_length=0.1, fc='k', ec='k')

    ax.set_xlim(0, 1)
    # Format
    plt.autoscale()
    fig.tight_layout()
    fig.set_size_inches(w=11,h=7)
