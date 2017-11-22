""" Useful helper plotting functions.  """

import pkg_resources

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import geopandas as gpd

PLATE_CARREE = ccrs.PlateCarree()
#: Nice defaults for CONUS projections
STATE_PROJ = ccrs.AlbersEqualArea(central_longitude=-97., central_latitude=38.)
STATE_EXTENT = [-121, -72, 22.5, 50]

#: Read and save a cached set of states. Hacky but works for our purposes
STATES_GEOJSON = None


def get_figsize(nrows=1, ncols=1, size=4., aspect=16./10.):
    """ Compute figure size tuple given columns, rows, size, and aspect. """
    width = ncols*size*aspect
    height = nrows*size
    return (width, height)


def _get_states():
    """ Load a cached Shapefile containing US states. """
    global STATES_GEOJSON
    if STATES_GEOJSON is None:
        STATES_GEOJSON = gpd.read_file(
            pkg_resources.resource_filename(
                'data/usa.geojson',
            )
        )
    else:
        return STATES_GEOJSON


def add_usa_states(ax, facecolor=cfeature.COLORS['land'],
                  edgecolor='k', **kwargs):
    """ Add a set of polygons to a given axis corresponding to the states
    in the USA. You *do not* need to pass a 'transform' argument to
    kwargs; that's already been taken care of here. """
    shp = _get_states()
    ax.add_geometries(shp.geometry, PLATE_CARREE,
                      facecolor=facecolor, edgecolor=edgecolor,
                      zorder=-9999, **kwargs)

    ax.set_extent(STATE_EXTENT)

    return ax


def usa_states_ax(ax=None, size=5., aspect=16./10.,
                  projection=None, frame=False,
                  facecolor=cfeature.COLORS['land'],
                  edgecolor='k', **kwargs):

    """ Create a GeoAxis of a specified size, with all the states in the USA
    superimposed as polygons"""
    if projection is None:
        projection = STATE_PROJ

    if ax is None:
        figsize = get_figsize(size=size, aspect=aspect)
        fig, ax = plt.subplots(1, 1, figsize=figsize, frameon=frame,
                               subplot_kw=dict(projection=projection,
                                               aspect='auto'))
    ax = add_usa_states(ax, facecolor, edgecolor, **kwargs)

    # Remove frame from axis
    ax.outline_patch.set_visible(False)

    return ax
