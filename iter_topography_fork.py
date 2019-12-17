from copy import deepcopy
from functools import partial
from itertools import cycle

import numpy as np

from mne.io.pick import channel_type, pick_types
from mne.utils import _clean_names, warn, _check_option, Bunch
from mne.channels.layout import _merge_grad_data, _pair_grad_sensors, find_layout
from mne.defaults import _handle_default
from mne.viz.utils import (_check_delayed_ssp, _get_color_list, _draw_proj_checkbox,
                    add_background_image, plt_show, _setup_vmin_vmax,
                    DraggableColorbar, _setup_ax_spines,
                    _check_cov, _plot_masked_image)

def _iter_topography(info, layout, on_pick, fig, fig_facecolor='k',
                     axis_facecolor='k', axis_spinecolor='k',
                     layout_scale=None, unified=False, img=False, axes=None,
                     legend=False, 
                     
                     y_scale = False, hide_xticklabels = True, hide_yticklabels = True):
    """Iterate over topography.

    Has the same parameters as iter_topography, plus:

    unified : bool
        If False (default), multiple matplotlib axes will be used.
        If True, a single axis will be constructed. The former is
        useful for custom plotting, the latter for speed.
    """
    from matplotlib import pyplot as plt, collections

    if fig is None:
        fig = plt.figure()

    def format_coord_unified(x, y, pos=None, ch_names=None):
        """Update status bar with channel name under cursor."""
        # find candidate channels (ones that are down and left from cursor)
        pdist = np.array([x, y]) - pos[:, :2]
        pind = np.where((pdist >= 0).all(axis=1))[0]
        if len(pind) > 0:
            # find the closest channel
            closest = pind[np.sum(pdist[pind, :]**2, axis=1).argmin()]
            # check whether we are inside its box
            in_box = (pdist[closest, :] < pos[closest, 2:]).all()
        else:
            in_box = False
        return (('%s (click to magnify)' % ch_names[closest]) if
                in_box else 'No channel here')

    def format_coord_multiaxis(x, y, ch_name=None):
        """Update status bar with channel name under cursor."""
        return '%s (click to magnify)' % ch_name

    fig.set_facecolor(fig_facecolor)
    if layout is None:
        layout = find_layout(info)

    if on_pick is not None:
        callback = partial(_plot_topo_onpick, show_func=on_pick)
        fig.canvas.mpl_connect('button_press_event', callback)

    pos = layout.pos.copy()
    if layout_scale:
        pos[:, :2] *= layout_scale
    if y_scale:
        pos[:,3] *= y_scale

    ch_names = _clean_names(info['ch_names'])
    iter_ch = [(x, y) for x, y in enumerate(layout.names) if y in ch_names]
    if unified:
        if axes is None:
            under_ax = plt.axes([0, 0, 1, 1])
            under_ax.axis('off')
        else:
            under_ax = axes
        under_ax.format_coord = partial(format_coord_unified, pos=pos,
                                        ch_names=layout.names)
        under_ax.set(xlim=[0, 1], ylim=[0, 1])

        axs = list()
    for idx, name in iter_ch:
        ch_idx = ch_names.index(name)
        if not unified:  # old, slow way
            ax = plt.axes(pos[idx])
            ax.patch.set_facecolor(axis_facecolor)
            for spine in ax.spines.values():
                spine.set_color(axis_spinecolor)
            if hide_xticklabels:
                ax.set(xticklabels=[])
            if hide_yticklabels:
                ax.set(yticklabels=[])
            if not legend:
                for tick in ax.get_xticklines() + ax.get_yticklines():
                    tick.set_visible(False)
            ax._mne_ch_name = name
            ax._mne_ch_idx = ch_idx
            ax._mne_ax_face_color = axis_facecolor
            ax.format_coord = partial(format_coord_multiaxis, ch_name=name)
            yield ax, ch_idx
        else:
            ax = Bunch(ax=under_ax, pos=pos[idx], data_lines=list(),
                       _mne_ch_name=name, _mne_ch_idx=ch_idx,
                       _mne_ax_face_color=axis_facecolor)
            axs.append(ax)
    if not unified and legend:
        ax = _legend_axis(pos)
        yield ax, -1
    
    if unified:
        under_ax._mne_axs = axs
        # Create a PolyCollection for the axis backgrounds
        verts = np.transpose([pos[:, :2],
                              pos[:, :2] + pos[:, 2:] * [1, 0],
                              pos[:, :2] + pos[:, 2:],
                              pos[:, :2] + pos[:, 2:] * [0, 1],
                              ], [1, 0, 2])
        if not img:
            under_ax.add_collection(collections.PolyCollection(
                verts, facecolor=axis_facecolor, edgecolor=axis_spinecolor,
                linewidth=1.))  # Not needed for image plots.
        for ax in axs:
            yield ax, ax._mne_ch_idx