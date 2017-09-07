import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import darpy as dr
import darpy.plot as dplt

PLATE_CARREE = ccrs.PlateCarree()
#: Nice defaults for CONUS projections
STATE_PROJ = ccrs.AlbersEqualArea(central_longitude=-97., central_latitude=38.)
STATE_EXTENT = [-121, -72, 22.5, 50]


def get_states():
    """ Load a cached Shapefile containing US states. """
    return dr.load_resource("usa.geojson")


def add_usa_states(ax, facecolor=cfeature.COLORS['land'],
                  edgecolor='k', **kwargs):

    shp = dr.load_resource("usa.geojson")
    ax.add_geometries(shp.geometry, PLATE_CARREE,
                      facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    ax.set_extent(STATE_EXTENT)

    return ax


def usa_states_ax(ax=None, size=5., aspect=16./10.,
                  projection=None, frame=False,
                  facecolor=cfeature.COLORS['land'],
                  edgecolor='k', **kwargs):

    if projection is None:
        projection = STATE_PROJ

    if ax is None:
        figsize = dplt.get_figsize(size=size, aspect=aspect)
        fig, ax = plt.subplots(1, 1, figsize=figsize, frameon=frame,
                               subplot_kw=dict(projection=projection,
                                               aspect='auto'))
    ax = add_usa_states(ax, facecolor, edgecolor, **kwargs)

    # Remove frame from axis
    ax.outline_patch.set_visible(False)

    return ax
