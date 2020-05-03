"""Useful functions that largely conform to the matplotlib interface."""
import math
from numbers import Real
from typing import Iterable, Optional

import numpy
from matplotlib import pyplot


def axes_iter(
        iterable: Iterable,
        num_cols: int = 3,
        width: Real = 13,
        height: Real = 4,
        tight_layout: bool = True,
        sharex: bool = False,
        sharey: bool = False,
        shared_ylabel: Optional[str] = None,
        clim_convex_hull: bool = False):
    """Iterate over the given iterable, but put a new set of axes on the context for each item.

    :param iterable: The elements over which to iterate.
    :param num_cols: The number of columns in the grid.
    :param width: The total width of the resulting grid in matplotlib units.
    :param height: The height of each row in matplotlib units.
    :param tight_layout: Iff true, call pyplot.tight_layout() after each plot.
    :param sharex: If specified, rows will share x-axes.
        This necessitates all axes being created in a single figure.
    :param sharey: If specified, each row will share y-axes.
    :param shared_ylabel: Iff sharey is true, and this is specified, set this as the common y-axis label.
    :param clim_convex_hull: Iff true, attempt to set the color scale limits to the convex hull of all internal color
        limits.
    """
    # All axes which have been yielded.
    all_axes = []

    # A temporary store of axes, from which we will yield.
    current_axes = []

    single_figure = sharex

    if single_figure:
        # If we are going to show in a single figure, figure out how many elements we have and pre-create axes.
        iterable = list(iterable)
        num_rows = int(math.ceil(len(iterable) / num_cols))
        _, new_axes = pyplot.subplots(num_rows, num_cols, figsize=(width, height * num_rows),
                                      sharex='all' if sharex else 'none', sharey='all' if sharey else 'none')
        current_axes = list(numpy.ravel(new_axes))

    for i, item in enumerate(iterable):
        first_item_in_row = i % num_cols == 0

        if len(current_axes) == 0:
            # Need to create another row of axes.
            _, new_axes = pyplot.subplots(1, num_cols, figsize=(width, height), sharey='row' if sharey else 'none')
            if num_cols == 1:
                # Have to work around matplotlib inconsistency here.
                current_axes = [new_axes]
            else:
                current_axes = list(new_axes)
        axes = current_axes.pop(0)
        pyplot.sca(axes)
        if first_item_in_row and sharey and shared_ylabel is not None:
            # We are sharing a y-axis, and want to set a common y-axis label. Set it here.
            pyplot.ylabel(shared_ylabel)
        yield item
        all_axes.append(axes)

        if tight_layout:
            pyplot.tight_layout()

    if clim_convex_hull:
        set_clim_to_convex_hull(*all_axes)


def set_clim_to_convex_hull(*all_axes):
    """Set the color scale limits of the given axes to the convex hull of all those specified."""
    min_limit = None
    max_limit = None

    # Save the current axes so we can restore the context after this function call.
    current_axes = pyplot.gca()

    for axes in all_axes:
        # This is the only public-API method for obtaining 'images' (e.g. a pcolormesh) from each axis.
        pyplot.sca(axes)
        image = pyplot.gci()
        this_min, this_max = image.get_clim()
        min_limit = this_min if min_limit is None else min(this_min, min_limit)
        max_limit = this_max if max_limit is None else max(this_max, max_limit)

    # We now have the new limits, so set everywhere.
    for axes in all_axes:
        pyplot.sca(axes)
        pyplot.clim(min_limit, max_limit)

    pyplot.sca(current_axes)
