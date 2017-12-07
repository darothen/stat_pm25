""" These transformers/mixins/utilities are designed to enable scikit-learn
workflows to easily work with xarray objects, even if they're multi-dimensional
or collections of data.

"""

from itertools import product
from functools import reduce
import warnings

from sklearn.base import TransformerMixin, clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import FeatureUnion

from tqdm import tqdm

import numpy as np
import xarray as xr

from .util import _months_surrounding, _isin, logger


class DatasetModel(object):
    """ Container class for automating the application of a statistical or
    machine learning pipeline to all the cells in a given Dataset.

    This class is designed to help automate the process of fitting a predictive
    model to all cells in a Dataset. It contains some rudimentary logic for
    subsetting the cells included in the analysis, as well as defining which
    variables are predictors and predictands in the passed Dataset. It's
    designed such that fitting transformations and predictive models can be
    easily done with a single click.

    Once a set of models has been fit, they can be used to predict an indicated
    field for another datset, provided that dataset has the same spatial
    grid and contains all the predictor variables. This class also helps to
    handle the pre/post-processing of data for prediction, since it keeps a
    record of how fields were de-meaned and scaled.

    """

    preprocessor = None

    def __init__(self, data, predictand, predictors,
                 grid_stack=['lat', 'lon'], cell_name='cell',
                 lon_range=None, lat_range=None, mask=None,
                 verbose=False):
        """ Initialize a grid of fitted models.

        Parameters
        ----------
        data : xarray.Dataset
            A dataset with dimensions ('time', 'lon', 'lat') and fields containing at least
            the set of predictors and predictand requested for fitting the model.
        predictand : str
            Name of predictand field
        predictors : list of str
            List of names of predictor fields
        grid_stack : list of str
            Dimensions to collapse to produce linear vector of feature sets
        cell_name : str
          Name of collapsed dimension containing feature sets
        {lon, lat}_range : tuple of (float, float) (optional)
            Min/max latitudes and longitudes to fit the model over (if less than
            the full domain)
        mask : xarray.DataArray
            A dataarray with dimensions ('lon', 'lat') corresponding to `data`, indicating
            which grid cells should be excluded from fitting. Cells with "True" will be
            skipped.
        verbose : logical
            Toggle progress bars and updates during modeling fitting/predicting

        """

        self._data = data
        self.verbose = verbose

        self.grid_stack = grid_stack
        self.cell_name = cell_name

        # Fit only a subset of the data?
        self.lon_range = lon_range
        self.lat_range = lat_range
        self._fit_subset = (lon_range is not None) and (lat_range is not None)
        self.mask = mask

        self.predictand = predictand
        self.predictors = predictors

        # Fitted model components
        self._grid_cell_results = []
        self._grid_cell_factors = []

        # Fit our pre-processor for later transformation
        if self.preprocessor is not None:
            if self.verbose > 0:
                print("Fitting pre-processor...")
            self._data = self.preprocessor.fit_transform(self._data)
        # self.to_model = self._data[self.predictors]

    @property
    def data(self):
        return self._data

    @property
    def grid_cell_factors(self):
        if not self._grid_cell_factors:
            raise ValueError("Must fit model before accessing grid cell"
                             " factors")
        return self._grid_cell_factors

    @property
    def grid_cell_results(self):
        if not self._grid_cell_results:
            raise ValueError("Must fit model before accessing grid cell"
                             " results")
        return self._grid_cell_results

    @property
    def _gcr_gcf_iter(self):
        """ Return an iterator over all the GridCellFactor and GridCellResult
        objects obtained from fitting the model to this data. """

        gcr_gcf_iter = zip(self.grid_cell_results, self.grid_cell_factors)
        if self.verbose:
            gcr_gcf_iter = tqdm(gcr_gcf_iter,
                                desc="Iterate over grid cell models",
                                total=len(self.grid_cell_results),
                                maxinterval=1.0)
        return gcr_gcf_iter

    @property
    def _ilon_ilat_iter(self):
        """ Return an iterator that produces the indices to access each
        latitude/longitude grid cell in the underlying dataset. """

        ilon_ilat_iter = product(range(len(self.data.lon)),
                                 range(len(self.data.lat)))
        if self.verbose:
            ilon_ilat_iter = tqdm(ilon_ilat_iter,
                                  desc="Loop over all grid cells",
                                  total=len(self.data.lon)*len(self.data.lat),
                                  maxinterval=1.0)
        return ilon_ilat_iter

    def cell_kernel(self, gcf):
        """ Using the information in the passed GridCellFactor, prepare
        data transformations and fit a model.

        This method should be prepared by the user; it's the only element
        that needs to be built to create a new machine learning model. It
        should return the original GridCellFactor passed to it, as well as
        a GridCellResult encapsulating the model fit for this GridCellFactor.

        """

        raise NotImplementedError

        local_selector = DatasetSelector(sel='isel', lon=gcf.ilon, lat=gcf.ilat)
        y = local_selector.fit_transform(self.data[self.predictand])

        # Short-circuit model fitting if we have no usable data
        if np.all(y.isnull()):
            return None, None

        # Create a transformer to produce the features we expect (local vs
        # hybrid synoptic), and produce a wrapped model object that implements
        # our cross-validation and feature selection strategy.
        transformer = make_transformer(self.predictors, gcf, self.hybrid)
        if self.cv is None:
            _model = SelectBestFeatures(
                self.estimator, transformer#, verbose=self.verbose
            )
        else:
            _model = SelectBestFeaturesWithCV(
                self.estimator, transformer, cv=dataset_yearly_loo_cv,
                #verbose=self.verbose
            )

        # Try to fit the model
        try:
            _model.fit(self.to_model, y)

            _score = _model.score(self.to_model, y)
            gcr = GridCellResult(_model, self.predictand, _model.features_, _score)
        except:
            gcr = None

        return gcr


    def _fit_cell(self, ilon, ilat):
        """ For a given latitude and longitude index in a dataset, fit the
        regression model. """

        if self._fit_subset:
            lon = float(self._data.isel(lon=ilon).lon.values)
            lat = float(self._data.isel(lat=ilat).lat.values)

            lon_lo, lon_hi = self.lon_range
            lat_lo, lat_hi = self.lat_range

            # Check bounds and short-circuit if necessary
            if not ((lon_lo <= lon <= lon_hi) and (lat_lo <= lat <= lat_hi)):
                return None, None

        gcf = GridCellFactor(ilon, ilat, self.dilon, self.dilat)

        # Skip masked datapoints
        if self.mask is not None:
            local_mask = self.mask.isel(lat=ilat, lon=ilon)
            if local_mask:
                return None, None

        # Fit the cell kernel if all is well
        try:
            gcr = self.cell_kernel(gcf)
        except:
            warnings.warn('Fit failed at {}'.format(gcf))
            gcr = None

        return gcf, gcr

    def fit_parallel(self, n_jobs=3, **kwargs):
        """ Similar to fit(), but using a parallel invocation. """
        import time
        from joblib import Parallel, delayed

        start = time.clock()
        results = Parallel(n_jobs=n_jobs, **kwargs)(
            delayed(self._fit_cell)(ilon, ilat)
            for ilon, ilat in self._ilon_ilat_iter
        )
        gcfs, gcrs = zip(*results)

        self._grid_cell_factors = list(gcfs)
        self._grid_cell_results = list(gcrs)

        elapsed = time.clock()
        elapsed = elapsed - start
        print("Time elapsed = ", elapsed)

    def fit(self):
        """ Fit the model for the given data. """

        for ilon, ilat in self._ilon_ilat_iter:
            gcf, gcr = self._fit_cell(ilon, ilat)
            self._grid_cell_factors.append(gcf)
            self._grid_cell_results.append(gcr)

    def get_result_stat(self, attr):
        """ Fetch a result statistic from the fitted model which has
        been saved with each GridCellResult.
        """

        # Construct a template/strawman Dataset for holding the attribute data
        attr_ds = self.data.copy().drop(self.predictors)
        ref_field = attr_ds[self.predictand].isel(time=0)
        attr_ds[attr] = (ref_field.dims, np.zeros_like(ref_field.values)*np.nan)
        _attr_vals = attr_ds[attr].values

        for gcr, gcf in self._gcr_gcf_iter:
            if gcr is None:
                continue

            _attr_vals[gcf.ilat, gcf.ilon] = getattr(gcr, attr)

        return attr_ds[[attr, ]]

    @property
    def score(self):
        # Construct a template/strawman Dataset for holding the attribute data
        attr_ds = self.data[[self.predictand, ]].copy()
        ref_field = attr_ds[self.predictand].isel(time=0)
        vals = ref_field.values

        for gcr, gcf in self._gcr_gcf_iter:
            if gcr is None:
                continue

            vals[gcf.ilat, gcf.ilon] = gcr.score
        ref_field.name = 'score'
        return ref_field

    def predict(self, X, preprocess=True):
        """ Predict new data using the fitted models.

        """

        if preprocess and (self.preprocessor is not None):
            if self.verbose > 0:
                print("Pre-processing data...")
            X = clone(self.preprocessor).fit_transform(X)

        # Drop fields which aren't in the list of predictors
        predicted = X[[self.predictand, ]].copy() * np.nan
        X = X[self.predictors]
        #_pred = predicted[self.predictand].values

        for gcr, gcf in self._gcr_gcf_iter:

            # If there was no result saved, then we should skip it (it means
            # that we probably masked out that grid cell from our analysis)
            if gcr is None:
                continue

            try:
                _y_pred = gcr.model.predict(X)
                _to_set = predicted[self.predictand].isel(lon=gcf.ilon, lat=gcf.ilat)
                _to_set[:] = _y_pred
            except:
                continue

        return predicted

    def to_pickle(self, filename):
        """ Serialize this model to disk for later evaluation. """
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class GridCellFactor(object):
    """ Simple wrapper class which caches the logic of subsetting
    from a gridded Dataset.

    """

    def __init__(self, ilon, ilat, dilon=None, dilat=None):

        self.ilon = ilon
        self.ilat = ilat
        self.dilon = dilon
        self.dilat = dilat

    @property
    def ilat_range(self):
        if self.dilat is None: return None
        lat_lo, lat_hi = self.ilat-self.dilat-1, self.ilat+self.dilat
        lat_lo = 0 if lat_lo < 0 else lat_lo
        return slice(lat_lo, lat_hi)

    @property
    def ilon_range(self):
        if self.dilon is None: return None
        lon_lo, lon_hi = self.ilon-self.dilon-1, self.ilon+self.dilon
        lon_lo = 0 if lon_lo < 0 else lon_lo
        return slice(lon_lo, lon_hi)

    def __repr__(self):
        return "GridCellFactor - [{:d}, {:d}]".format(self.ilon, self.ilat)


class GridCellResult(object):
    """ Simple wrapper class which caches the results of fitting
    an sklearn [ipeline/model for a given grid cell

    """

    def __init__(self, model, predictand, predictors, score=None, **kwargs):

        self.model = model
        self.predictand = predictand
        self.predictors = predictors
        self.score = score

        # Set any other keyword arguments as attributes
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __repr__(self):
        base =  "GridCellResult - {} ~ {}".format(
            self.predictand, " + ".join(self.predictors)
        )
        if self.score is not None:
            stat = "[{:g}]".format(self.score)
            base = base + " " + stat
        return base



def dataset_yearly_loo_cv(X, y=None, dim='time'):
    """ Generate training/testing index splits by iterating over the
    years present in the timestamp observations for a Dataset.

    For example, if you have 12 years of monthly output (for a total of
    144 indices along your time dimension), this cross-validator will
    create 12 folds, each with 12 months of training data and 132 months
    of testing data.

    Parameters
    ----------
    X : Dataset
        A Datset containing feature vectors as unique Variables, aligned on
        common axes
    y : np.array (optional; not used)
        A Vector of predictands corresponding to each observation in the Dataset
    dim : str (optional; default='time')
        Dimension corresponding to time values; should be the observational
        dimension in `X`

    """

    all_years = X[dim].dt.year.values
    years = np.unique(all_years)
    inds = np.arange(len(all_years))
    for year in years:
        test_inds = inds[all_years == year]
        train_inds = inds[all_years != year]
        yield train_inds, test_inds


class NaiveYearlyLeaveOneOut(BaseCrossValidator):
    """ Cross-validate a dataset by taking sequential chunks of length `n`. """

    def __init__(self, n=3):
        self.n = 3

    def _iter_test_indices(self, X, y=None, groups=None):
        inds = np.arange(self.n)
        for i in range(len(X) // self.n):
            yield inds + self.n * i

    def get_n_splits(self, X, y=None, groups=None):
        return len(X) / self.n


class NoFitMixin(object):
    """ Specialized mixin which automatically defines a pass-through
    `fit` function which requires no special operations. """

    def fit(self, X, y=None):
        """ Pass-through without performing any actual fitting. """
        return self


class Reshaper(object):
    """ Helper tool to help re-shape DataArrays and Datasets via stacking
    and unstacking.

    Note that unlike the normal Data{Array,set}.stack() function, this utility
    automatically moves the raveled dimension to the front of the object instead
    of keeping it at the end.

    Parameters
    ----------
    dims_to_stack : list of str
        The dimension names to collapse / ravel

    stacked_name : str
        Name of collapsed / raveled dimension

    """

    def __init__(self, dims_to_stack=['lat', 'lon'], stacked_name='cell'):
        self.dims_to_stack = dims_to_stack
        self.stacked_name = stacked_name

    def stack(self, ds):
        """ Stack the given dataset based on the saved configuration. """
        stacked = ds.stack(**{self.stacked_name: self.dims_to_stack})
        # Move stacked dim to front
        return stacked.T

    def unstack(self, ds):
        """ Un-stack a previously stacked dataset. """
        return ds.unstack(self.stacked_name)


class DatasetFeatureUnion(FeatureUnion):
    """ FeatureUnion for xarray Datasets.

    This estimator sequentially applies a list of transformers and then
    concatenates the results into a Dataset. The transformers should all
    consume and produce Datasets themselves.

    See sklearn.pipeline.FeatureUnion for more information and examples.

    """

    def __init__(self, transformer_list, **kwargs):
        self.transformer_list = transformer_list
        super().__init__(transformer_list, **kwargs)

    #@logger
    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    #@logger
    def fit_transform(self, X, y=None):
        if y is None:
            return self.fit(X).transform(X)
        else:
            return self.fit(X, y).transform(X)

    #@logger
    def transform(self, X):
        # Assumes X is a Dataset
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: xr.merge([X1, X2]), Xts)
        return Xunion


class FieldExtractor(TransformerMixin, NoFitMixin):
    """ Subset fields from a Dataset. """

    def __init__(self, fields):
        self.fields = fields

    def transform(self, X):
        return X[self.fields]


class MonthSelector(TransformerMixin, NoFitMixin):
    """ Select timesteps corresponding to and surrounding a given month
    from a dataset. """

    def __init__(self, month, width=1, dim='time'):
        self.month = month
        self.dim = dim
        self._months = \
            _months_surrounding(self.month) if width else [self.month, ]

    #@logger
    def transform(self, X):
        idx = _isin(X['{}.month'.format(self.dim)], self._months)
        return X.sel(**{self.dim: idx})


class DatasetSelector(TransformerMixin, NoFitMixin):
    """ Apply dimensional indexing/selecting to a Dataset.

    Note that you can control whether or not to use index-based or label-based
    selection via the 'sel' argument. For example:

    >>> selector = DatasetSelector(sel='sel', lon=slice(20, 40), lat=10)
    >>> selector.transform(my_data)

    is functionally equivalent to the traditional selection semantics

    >>> my_data.sel(lon=slice(20, 40), lat=10)

    """

    def __init__(self, sel='sel', **indexers):
        self.sel = sel
        self.indexers = indexers

    #@logger
    def transform(self, X, **kwargs):
        if self.sel == 'sel':
            return X.sel(**self.indexers)
        else:
            return X.isel(**self.indexers)


class Normalizer(TransformerMixin):
    """ Normalize and scale each field in a Dataset, caching the mean and
    standard deviation to recover the original data if desired.

    By default, the given Dataset is both centered around zero and scaled to have
    unit variance. Either option can be disabled via the `with_mean` and
    `with_std` arguments.

    """

    def __init__(self, copy=True, with_mean=True, with_std=True, dim='time'):
        self.dim = dim
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None

    #@logger
    def fit(self, X, Y=None, **kwargs):
        if self.with_mean:
            self.mean_ = X.mean(self.dim, skipna=True)
        if self.with_std:
            self.scale_ = X.std(self.dim, skipna=True)
        return self

    #@logger
    def transform(self, X, **kwargs):
        if self.copy:
            X = X.copy()
        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.scale_
        return X

    #@logger
    def inverse_transform(self, X, copy=None):
        copy = copy if copy is not None else self.copy
        if copy:
            X = X.copy()
        if self.with_std:
            X *= self.scale_
        if self.with_mean:
            X += self.mean_
        return X


class Stacker(TransformerMixin, NoFitMixin):
    """ Transform a Dataset or DataArray by stacking the given dimensions. """

    def __init__(self, dims_to_stack=['lat', 'lon'], stacked_name='cell'):
        self.dims_to_stack = dims_to_stack
        self.stacked_name = stacked_name

    #@logger
    def transform(self, X):
        stacked = X.stack(**{self.stacked_name: self.dims_to_stack})
        # Move stacked dim to front
        return stacked.T

    #@logger
    def inverse_transform(self, X):
        return X.unstack(self.stacked_name)


class YearlyMovingAverageDetrender(TransformerMixin):
    """ Break apart a Dataset or DataArray into a set of values for a given month,
    and remove the local moving average from them. """

    def __init__(self, dim='time', with_mean=False, n_years=5, center=True,
                 min_periods=1, copy=True):
        self.dim = dim
        self.with_mean = with_mean
        self.n_years = n_years
        self.center = center
        self.min_periods = min_periods
        self.copy = copy

        self.moving_avg_trend_ = None

    #@logger
    def fit(self, X, y=None, **kwargs):
        Xg = X.groupby('time.month')
        if self.with_mean:
            # We have to include the mean here as an offset so it gets computed for each
            # monthly timeseries of annual values
            self.moving_avg_trend_ = Xg.apply(
                lambda x: x.rolling(center=self.center,
                                    min_periods=self.min_periods,
                                    **{self.dim: self.n_years}).mean() - x.mean(
                    self.dim, skipna=True)
            )
        else:
            self.moving_avg_trend_ = Xg.apply(
                lambda x: x.rolling(center=self.center,
                                    min_periods=self.min_periods,
                                    **{self.dim: self.n_years}).mean()
            )

        return self

    #@logger
    def transform(self, X, **kwargs):
        # Xg = X.groupby('time.month')
        X_detrend = X - self.moving_avg_trend_
        return X_detrend.dropna(self.dim, how='all')


class DatasetFunctionTransformer(TransformerMixin, NoFitMixin):
    """ Apply an arbitrary function to each field in a Dataset. """

    def __init__(self, func, copy=False):
        self.func = func
        self.copy = copy

    def transform(self, X):
        _X = X if not self.copy else X.copy()
        return _X.apply(self.func)


class DatasetAdapter(TransformerMixin, NoFitMixin):
    """ Mutate the given fields into a Dataset into a 2D array such that
    each field is contained in a separate column. """

    def __init__(self, fields=[], drop=[]):
        self.fields = fields
        self.drop = drop

    # @logger
    def transform(self, X):
        _X = X.copy()

        if self.drop:
            _X = _X.drop(self.drop)

        _X = _X.to_dataframe()

        if self.fields:
            _X = _X[self.fields]

        return _X.values
