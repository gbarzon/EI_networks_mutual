import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

import seaborn as sns

colors = ("#76c893", "#184e77")
labels = ('E', 'I')
folder_figures = 'figures/'
format_fig = '.svg'
my_cmap_discrete = ListedColormap(sns.color_palette('mako_r').as_hex())
my_cmap_continuous = sns.color_palette('mako_r', as_cmap=True)

def load_matplotlib_local_fonts():
    from matplotlib import font_manager
    
    font_path = '/home/barzon/Avenir.ttc'
    
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    #  Set it as default matplotlib font
    plt.rcParams.update({
        'font.sans-serif': prop.get_name(),
    })
    
#load_matplotlib_local_fonts()
plt.rcParams.update({'font.size': 16})

def plot_simulation(states, inputs, hs, dt,
                    t_min = 0, bins = 100, max_steps_to_plot = int(1e5),
                    figsize = (8,3), height_ratios = [0.3, 1], fname = None):

    steps = states.shape[0]
    if steps < max_steps_to_plot:
        max_steps_to_plot = steps
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, height_ratios=height_ratios)

    ax = fig.add_subplot(gs[0,0])
    ax.plot(np.arange(max_steps_to_plot)*dt, hs[inputs][-max_steps_to_plot:], c='k')
    #ax.set_ylabel('Input')
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1,0])
    for idx, state in enumerate(states.T):
        ax.plot(np.arange(max_steps_to_plot)*dt, state[-max_steps_to_plot:], label=labels[idx], c=colors[idx] )
    ax.set_xlabel('t')
    ax.set_ylabel('x(t)')
    ax.legend()
    ax.spines[['top','right']].set_visible(False)

    ax = fig.add_subplot(gs[:,1])
    sns.histplot(x=states[t_min:,0], y=states[t_min:,1], bins=bins, ax=ax, cmap='mako_r', stat='density', cbar=True)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    
    if fname is not None:
        plt.savefig(folder_figures+fname+format_fig, transparent=True)
        
    plt.show()