import numpy as np
import matplotlib.pyplot as plt

from dirdet.config.physics import SourceConfig, SignalType, NeutrinoRegistry, NeutrinoGroup, WIMP_PLOT_LINES
from dirdet.helpers import latex_float

#===============================================================#
#------------------------ WIMP Plotting ------------------------#
#===============================================================#
def plot_wimp_recoil(
        ax,
        source: SourceConfig,
        x_data: np.ndarray,
        y_data: np.ndarray,
        ls_index: int, # required for line style index
        colour: str = "grey"
) -> None:
    
    linestyle = WIMP_PLOT_LINES.WIMP_LINES[ls_index]
    m_label = rf"$m_{{\chi}}={source.equiv_wimp_mass}$ GeV"
    base, exp = f"{source.equiv_wimp_sigma:.2e}".split("e")
    sig_label = rf"${base} \times 10^{{{int(exp)}}}$ cm$^{{2}}$"

    label = f"{m_label},\n{sig_label}"
    ax.loglog(x_data, y_data, label=label, color=colour, linestyle=linestyle)

    return None
    

#===================================================================#
#------------------------ Neutrino Plotting ------------------------#
#===================================================================#

def plot_neutrino_flavour(
        ax, # axes object
        source: SourceConfig,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        how: str = "loglog",
        fill: bool = True,
        use_source_col: bool = True,
        manual_label: str | bool = False,
        fill_alpha: float = 0.2
) -> None:
    

    # extract the label and colour, using the method
    label = source.label if not manual_label else manual_label
    colour = source.color if use_source_col else None
    
    # method for plotting
    method = {
        "loglog": ax.loglog,
        "semilogx": ax.semilogx,
        "semilogy": ax.semilogy,
        "plot": ax.plot
    }

    fill_lower = {
        "loglog": 1e-10,
        "semilogy": 1e-10,
        "semilogx": 0,
        "plot": 0,
    }
    plot_fn = method[how]
    # discrete check
    if not isinstance(y_arr, np.ndarray):
        ax.loglog([x_arr,x_arr], [1e-2,y_arr,], color=colour, label=label)
        return None

        # sort x_arr before plotting
    
    sort_idx = np.argsort(x_arr)
    x_arr = x_arr[sort_idx]
    y_arr = y_arr[sort_idx]

    # fill colour between
    line, = plot_fn(x_arr, y_arr, label=label, color=colour)
    fill_colour = line.get_color()

    if fill:    ax.fill_between(x_arr, y_arr , fill_lower[how], color=fill_colour, alpha=fill_alpha)

    return None


def plot_neutrino_groups(
        x_data: dict | np.ndarray, 
        y_data_dict: dict, 
        figsize: tuple[int,int] = (12, 8),
        sup_xaxis: str | None = None,
        sup_yaxis: str | None = None,
        sup_title: str | None = None,
        xlim: tuple = (5*1e-2,5*1e1), 
        ylim = (1e-1,1e12),
) -> None:
    ''' Plots the x and y data for neutrino data''' 

    all_sources = NeutrinoRegistry.all_sources()
    unique_groups = list(NeutrinoGroup)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    if sup_xaxis is not None:   fig.supxlabel(sup_xaxis)
    if sup_yaxis is not None:   fig.supxlabel(sup_yaxis)
    if sup_title is not None:   fig.supxlabel(sup_title)

    for i, group_enum in enumerate(unique_groups):
        ax = axes[i]
        ax.set_title(f"{group_enum.name} Neutrinos")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Filter sources belonging to this group
        group_sources = [s for s in all_sources if s.group == group_enum]
        
        for source in group_sources:
            # Use the source.name to fetch data from your dicts
            y_val = y_data_dict[source.name]
            x_val = x_data[source.name] if isinstance(x_data, dict) else x_data
            
            # Call the refactored flavour plot
            plot_neutrino_flavour(ax=ax, source=source, x_arr=x_val, y_arr=y_val)

        ax.legend(loc='best', fontsize='small')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()