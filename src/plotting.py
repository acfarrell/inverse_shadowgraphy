import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# define a handy function for wiping the ticks and labeles off of plot axes
def no_ticks(ax=None):
    if ax is None:
        ax = plt.gca()
    return ax.tick_params(left=False,
                          bottom=False,
                          labelleft=False,
                          labelbottom=False)
# and another for setting the axis limits to stay where they are when plotting multiple things
def freeze_ax(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    return

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar