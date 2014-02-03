from __future__ import division
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from math import sqrt, ceil
import networkx as nx

def plot_all(images):
    n = len(images)
    grid_dims = int(ceil(sqrt(n)))
    if n <= grid_dims * (grid_dims-1):
        vdim = grid_dims-1
    else:
        vdim = grid_dims
    fig = plt.figure()
    i = 0
    while i < n:
        gx = int(i//grid_dims)
        gy = int(i%grid_dims)
        ax = fig.add_subplot(vdim, grid_dims, i+1)
        plot_tuple(ax, images[i])
        i += 1

def plot_tuple(ax, tpl):
    import numpy as np
    if isinstance(tpl, np.ndarray):
        ax.imshow(tpl, interpolation='nearest', cmap=cm.Greys_r)
    else:
        try:
            img, graph = tpl
            ax.imshow(img, interpolation='nearest', cmap=cm.Greys_r)
            nx.draw_networkx_nodes(graph, {node:(node[1],node[0]) for node in
                graph.nodes()}, node_size=4)
            nx.draw_networkx_edges(graph, {node:(node[1],node[0]) for node in
                graph.nodes()}, edge_color='b')
        except ValueError:
            ax.imshow(tpl, interpolation='nearest', cmap=cm.Greys_r)



def plotgrid(rows, cols, title=None, limit=None):
    """Returns a grid of plots on a figure.
    You can show an image as follows.

    >>> img = [[0,1],[1,0]]
    >>> plots = plotgrid(5, 4) # 5 rows, 4 columns
    >>> plots[3][2].imshow(img, interpolation='nearest')

    This will show an image in the 4th row and 3rd column.
    """
    fig = plt.figure()
    if limit is None:
        limit = rows * cols
    if title:
        plt.figtext(0.5, 0.965, title, ha='center', color='black',
                weight='bold', size='large')
    plots = [[
        fig.add_subplot(rows, cols, j*cols + i + 1)
            for i in range(cols) if i + j * cols < limit]
                for j in range(rows)]
    return plots


class IndexTracker:
    def __init__(self, ax, plots, figure):
        self.plots = plots
        self.ax = ax
        self.index = 0
        self.figure = figure
        ax.set_title('use scroll wheel to navigate images')

        self.update()

    def onscroll(self, event):
        if event.key=='a':
            self.index -= 1
        elif event.key=='d':
            self.index += 1
        self.index %= len(self.plots)
        self.update()

    def update(self):
        print("updating to ", self.index)
        self.ax.clear()
        plot_tuple(self.ax, self.plots[self.index])
        self.figure.canvas.draw()

def frame_by_frame(plots):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    tracker = IndexTracker(ax, plots, fig)
    fig.canvas.mpl_connect('key_press_event', tracker.onscroll)
    fig._tracker = tracker
    return tracker

