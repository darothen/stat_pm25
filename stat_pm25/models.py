""" This module encodes logic and tools for reproducing the statistical
model developed in `Shen et al (2017) `_. The code herein stems from work
in notebook 09_lu_shen_reproduction, but generalizes things so that it's a
little bit easier to preprocess, fit, and evaluate a model given a model
or observational dataset.

.. _Shen et al (2017): www.atmos-chem-phys.net/17/4355/2017/

"""

from itertools import product, chain, combinations
from collections import namedtuple

from scipy.stats import pearsonr, spearmanr
from sklearn.utils.extmath import svd_flip
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

from .util import (stack_fields, _clean_xy, logger)
from .sklearn import (DatasetSelector, MonthSelector,
                      YearlyMovingAverageDetrender, FieldExtractor, Normalizer,
                      Stacker, DatasetAdapter, DatasetFeatureUnion,
                      dataset_yearly_loo_cv, GridCellFactor, GridCellResult,
                      DatasetModel)

#: Pseudo-class for quickly evaluating timeseries fits at given locations
Site = namedtuple("Site", ['name', 'lon', 'lat'])
sites = [
    Site('Georgia', -82.5, 32.5),
    Site('Boston', -71., 42),
    Site('Louisville', -85.76, 38.3)
]
sites = {site.name: site for site in sites}

def make_transformer(predictors, gcf, hybrid=False):
    """ Generate the transformation pipeline for use in the Shen et al (2017)
    model fitting. """
    local_selector = DatasetSelector(sel='isel', lon=gcf.ilon, lat=gcf.ilat)
    area_selector = DatasetSelector(sel='isel',
                                    lon=gcf.ilon_range, lat=gcf.ilat_range)

    #: Local transform - Extract local (in-grid-cell) timeseries for
    #                    regression
    local_pipeline = Pipeline([
        ('predictors', FieldExtractor(predictors)),
        ('normalize', Normalizer()),
        ('subset_latlon', local_selector),
    ])

    if not hybrid:
        return local_pipeline

    else:
        #: Hybrid transformer - Compute the SVD-compressed local fields
        #                       and derive timeseries of them at the grid cell
        hybrid_pipeline = Pipeline([
            ('predictors', FieldExtractor(predictors)),
            ('subset_latlon', area_selector),
            ('normalize', Normalizer()),
            ('stack', Stacker(['lon', 'lat'], 'cell')),
            ('svd_modes', SVDModeCompressor(predictors, max_modes=2)),
        ])

        # Merge the local/hybrid transformers together
        combined_local_hybrid = Pipeline([
            ('features', DatasetFeatureUnion([
                ('local', local_pipeline),
                ('hybrid', hybrid_pipeline)
            ])),
        ])

        return combined_local_hybrid


class SVDModeCompressor(TransformerMixin):
    """ Compresses multi-dimensional fields into principal component timeseries.

    Given a multi-dimensional set of fields of data `X` and a corresponding
    timeseries `y`, compute the pairwise correlation between each cell for each
    field in `X` and perform a singular value decomposition to compress the
    result multi-dimensional correlation fields. Using this SVD, then generate
    the set of principal component timeseries corresponding to each orthogonal
    mode. Prior to using this transformer, you should re-shape or stack your
    data such that all the non-time coordinate dimensions (e.g. lat/lon) are
    collapsed in a signal coordinate and the resulting data can be converted into
    a 2D matrix.

    Parameters
    ----------
    predictors : list of str
        The list of fields in `X` to consider when generating the correlation
        tensors

    predictand : str (defaults to `None`)
        A field in `y` which is the regressand timeseries. In most cases, it's
        best to ensure `y` is simply array-like (e.g. a NumPy array or a
        DataArray) and leave this as None.

    max_modes : int
        Maximum number of principal component timeseries to produce


    Attributes
    ----------
    F : Stacked correlation matrices between each cell/predictor in `X` and `y`
    u : SVD weights
    v : Variable weights
    evr : Explained variance ratio for each mode

    """

    def __init__(self, predictors, predictand=None, max_modes=2):
        self.predictors = predictors
        self.predictand = predictand
        self.max_modes = max_modes

        self.pt_corrs_ = None
        self.svd_modes_ = None

    ##@logger
    def fit(self, X, y=None):

        # Compute field correlations
        if self.predictand:
            yf = y[self.predictand].values
        else:
            yf = y.copy()

        self.pt_corrs_ = X.copy().drop(self.predictors + ['time',])
        field_rs = []
        for field in self.predictors:
            xf = X[field].values
            rs, ps = _calc_correlations(yf, xf)
            field_rs.append(rs)
            self.pt_corrs_[field + '_r'] = (['cell', ], rs)
            self.pt_corrs_[field + '_p'] = (['cell', ], ps)

        # Create SVD modes
        # self.F = stack_fields(self.pt_corrs,
        #                       fields=[p+"_r" for p in self.predictors],
        #                       reshape=False)
        self.F = np.stack([rs for rs in field_rs], axis=-1)
        self.F[np.isnan(self.F)] = 0.
        self.u, self.v, self.evr = _calc_svd(self.F)

        # Some legacy code archiving which part of the SVD decomp is what
        self.svd_modes_ = self.pt_corrs_.copy().drop(
            [p+"_r" for p in self.predictors] + [p+"_p" for p in self.predictors]
        )
        self.svd_modes_['field'] = (['field', ], self.predictors)
        self.svd_modes_['mode'] = (['mode', ], range(1, len(self.predictors)+1))
        self.svd_modes_['svd_weights'] = (['cell', 'mode'], self.u)
        self.svd_modes_['var_weights'] = (['mode', 'field'], self.v)
        self.svd_modes_['explained_variance'] = (['mode', ], self.evr)

        return self


    ##@logger
    def transform(self, X):

        Sks = []
        for ti in range(len(X.time)):
            # Compute M matrix
            _ds = X.isel(time=ti)
            M = stack_fields(_ds, self.predictors, reshape=False)

            Ski = np.inner(np.dot(self.u.T, M), self.v).diagonal()
            Sks.append(Ski)
        Sks = np.asarray(Sks)

        # Convert to a Dataset
        Sks_ds = X.copy().drop(self.predictors)
        Sis = []
        for i in range(1, self.max_modes+1):
            Si = 'S_{:d}'.format(i)
            Sks_ds[Si] = (['time', ], Sks[:, i-1])
            Sis.append(Si)
        Sks_ds = Sks_ds[Sis]

        return Sks_ds


class SelectBestFeatures(BaseEstimator):
    def __init__(self, estimator, transformer,
                 dim='time', n_features=3, verbose=0):
        self.estimator = estimator
        self.transformer = transformer

        self.dim = dim
        self.n_features = n_features
        self.verbose = verbose

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    ##@logger
    def fit(self, X, y):
        """ Note that X should be a Dataset which we can sample along
        the dimension 'dim'; y should be a vector with the same length
        as that dimension and corresponding to those values """

        _transformer = clone(self.transformer).fit(X, y)
        Xt = _transformer.transform(X)

        all_features = Xt.data_vars
        self.all_combos_ = list(
            chain(*[combinations(all_features, i)
                    for i in range(1, 1 + self.n_features)])
        )
        if self.verbose > 0:
            combo_iter = tqdm(self.all_combos_)
        else:
            combo_iter = self.all_combos_

        self.all_preds_ = []
        for feats in combo_iter:
            feats = list(feats)
            _estimator = clone(self.estimator)
            _estimator.fit(Xt[feats], y)
            y_pred = _estimator.predict(Xt[feats])
            self.all_preds_.append(y_pred)
        self.all_preds_ = np.asarray(self.all_preds_)

        # score = r2_score((y_true, y_pred)
        # score = pearsonr(y_true, y_pred)[0]**2
        self.scores_ = [pearsonr(y, self.all_preds_[i])[0]  # **2
                        for i in range(len(self.all_combos_))]
        # Recover the best model
        idx = np.argmax(self.scores_)
        self.features_ = list(self.all_combos_[idx])
        self.features_idx_ = idx

        self.transformer_ = clone(self.transformer).fit(X, y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self.transform(X), y)

        return self

    ##@logger
    def transform(self, X):
        return self.transformer_.transform(X)[self.features_]

    ##@logger
    def predict(self, X):
        transformed = self.transform(X)
        return self.estimator_.predict(transformed)

    ##@logger
    def score(self, X, y):
        transformed = self.transform(X)
        return self.estimator_.score(transformed, y)

class SelectBestFeaturesWithCV(BaseEstimator):
    """ This is an adaptation of BestFeatureCombo which works
    with a Dataset for X and a vector for y, as well as cross-validators
    which also work on those types of data.

    Because we anticipate generating features as part of the pipeline, we
    don't know a priori how many different combinations of features we'll
    need to test. We might be able to figure this out by reversing the order
    of our cross-validation and feature selection loops, but then we run into
    other issues with respect to picking the best features at the end. So,
    for now, we require the user to think about how many features their
    pipeline may generate and tell us up front.

    """

    def __init__(self, estimator, transformer, cv,
                 dim='time', n_features=3, verbose=0):
        self.estimator = estimator
        self.transformer = transformer
        self.cv = cv

        self.dim = dim
        self.n_features = n_features
        self.verbose = verbose

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    ##@logger
    def fit(self, X, y):
        """ Note that X should be a Dataset which we can sample along
        the dimension 'dim'; y should be a vector with the same length
        as that dimension and corresponding to those values """

        y = np.asarray(y)

        self.all_combos_ = None
        pred_rows = []
        cv_count = 1
        for train_inds, test_inds in self.cv(X, y, self.dim):
            if self.verbose > 0:
                print("CV split:", cv_count)
            _X_train = X.isel(**{self.dim: train_inds})
            _X_test = X.isel(**{self.dim: test_inds})
            _y_train = y[train_inds]
            _y_test = y[test_inds]

            n_obs = len(_X_train[self.dim])

            # Transform step - generate features
            _transformer = clone(self.transformer).fit(_X_train, _y_train)
            _X_train_trans = _transformer.transform(_X_train)
            _X_test_trans = _transformer.transform(_X_test)

            # Set up iteration over features
            if self.all_combos_ is None:
                all_features = _X_train_trans.data_vars
                n_feat = len(all_features)
                self.all_combos_ = list(chain(*[combinations(all_features, i)
                                                for i in
                                                range(1, 1 + self.n_features)]))
            if self.verbose > 0:
                combo_iter = tqdm(self.all_combos_)
            else:
                combo_iter = self.all_combos_

            # Iterate over all features, making a test prediction
            pred_cols = []
            for feats in combo_iter:
                feats = list(feats)
                _estimator = clone(self.estimator)
                _estimator.fit(_X_train_trans[feats], _y_train)
                _y_pred = _estimator.predict(_X_test_trans[feats])

                pred_cols.append(_y_pred)

            pred_rows.append(np.asarray(pred_cols))
            cv_count += 1

        # Process all our saved predictions to assess best candidate
        self.all_preds_ = np.concatenate(pred_rows, axis=-1).T

        # score = r2_score((y_true, y_pred)
        # score = pearsonr(y_true, y_pred)[0]**2
        self.scores_ = [pearsonr(y, self.all_preds_[:, i])[0]  # **2
                        for i in range(len(self.all_combos_))]

        # Recover the best model
        idx = np.argmax(self.scores_)
        self.features_ = list(self.all_combos_[idx])
        self.features_idx_ = idx

        self.transformer_ = clone(self.transformer).fit(X, y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self.transform(X), y)

        return self

    ##@logger
    def transform(self, X):
        return self.transformer_.transform(X)[self.features_]

    ##@logger
    def predict(self, X):
        transformed = self.transform(X)
        return self.estimator_.predict(transformed)

    ##@logger
    def score(self, X, y):
        transformed = self.transform(X)
        return self.estimator_.score(transformed, y)


def _calc_correlations(y, x, r='pearson'):
    """ Compute pairwise correlations between a reference timeseries (y) and
    a spatial timeseries in a DataArray (x).

    This version differs from the reproduction notebook in that we assume
    that the user has sanitized the data before passing to the correlation
    coefficient. This includes stacking dimensions.

    Parameters
    ----------
    y : 1D array-like
        Array with shape (n, )
    x : 2D array-like
        Array with shape (v, n)

    Returns
    -------
    rs, ps : 1D array-like
        Correlation coefficients and two-tailed P-values, in arrays with
        shape (v, )

    """
    if r == 'pearson':
        r_func = pearsonr
    elif r == 'spearman':
        r_fund = spearmanr
    else:
        raise ValueError("No correlation function '{}'".format(r))

    # Calculate correlations
    # TODO: could probably iterate directly over x instead of its indices
    rr = np.asarray([spearmanr(*_clean_xy(x[i], y))
                     for i in range(len(x))])
    rs, ps = rr[:, 0], rr[:, 1]

    return rs, ps

def _calc_svd(F):
    """ Compute the singular-value decomposition for a given set of
    geospatial fields.

    Parameters
    ----------
    F : 2D array-like
        An array of shape (p, m), where "p" is the total number of grid cells
        in a spatial field (e.g. nlat*nlon) and "m" is the number of fields
        used in the SVD compression

    Returns
    -------
    u : 2D array-like (m, p)
        spatial weights
    v : 2D array-like (m, m)
        field/variable weights
    evr : 1D array-like (m, )
        explained variance ratio for each mode

    """

    # De-mean data along each row to improve numerical stability
    F = F - np.mean(F, axis=0)
    u, s, v = np.linalg.svd(F, full_matrices=False)

    # Adjust columns of u and rows of v such that the loadings in the columns
    # in u that are the alrgest in absolute value are always positive
    u, v = svd_flip(u, v)

    # Compute explained variance fraction for each mode
    expl_var = (s**2) / len(s)
    total_var = np.sum(expl_var)
    evr = expl_var / total_var

    return u, v, evr


class SimpleRegress(DatasetModel):

    def __init__(self, *args, month=6, **kwargs):
        self.month = month
        self.dilat = self.dilon = 0

        self.preprocessor = Pipeline([
            ('subset_time', MonthSelector(self.month)),
            ('detrend', YearlyMovingAverageDetrender())
        ])

        super().__init__(*args, **kwargs)

    def cell_kernel(self, gcf):
        """ Using the information in the passed GridCellFactor, prepare
        data transformations and fit a model.

        This method should be prepared by the user; it's the only element
        that needs to be built to create a new machine learning model. It
        should return the original GridCellFactor passed to it, as well as
        a GridCellResult encapsulating the model fit for this GridCellFactor.

        """

        local_selector = DatasetSelector(sel='isel', lon=gcf.ilon,
                                         lat=gcf.ilat)
        y = local_selector.fit_transform(self.data[self.predictand])

        # Short-circuit model fitting if we have no usable data
        # if np.all(y.isnull()):
        #     return None, None

        _model = Pipeline([
            ('subset_latlon', DatasetSelector(
                sel='isel', lon=gcf.ilon, lat=gcf.ilat)
            ),
            ('predictors', FieldExtractor(self.predictors)),
            ('normalize', Normalizer()),
            ('dataset_to_array', DatasetAdapter(
                drop=['lat', 'lon'])),
            ('pca', PCA(n_components=3)),
            ('linear', LinearRegression()),
        ])

        try:
            print(gcf, end=" ")
            _model.fit(self.data, y)

            _score = _model.score(self.data, y)
            print(_score)
            gcr = GridCellResult(_model, self.predictand, self.predictors, _score)
        except:
            print("FAIL")
            gcr = None

        return gcr


class OldShen2017Model(object):
    """ Encapsulation of predictive model derived by Shen et al (2017).

    This class is designed to help automate the process of fitting the Shen et
    al predictive model. It's specialized to handle fitting for:

        - 1 specific month of the year
        - one dataset

    Once a set of models has been fit, they can be used to predict an indicated
    field for another datset, provided that dataset has the same spatial
    grid and contains all the predictor variables. This class also helps to
    handle the pre/post-processing of data for prediction, since it keeps a
    record of how fields were de-meaned and scaled.

    """

    def __init__(self, data,
                 month=1, dilon=6, dilat=4,
                 predictand='PM25',
                 predictors=['TEMP', 'RH', 'PRECIP', 'U', 'V'],
                 lon_range=None, lat_range=None, mask=None,
                 hybrid=False, n_predictors=3, cv=None,
                 verbose=False):
        """ Initialize a grid of fitted models.

        Parameters
        ----------
        data : xarray.Dataset
            A dataset with dimensions ('time', 'lon', 'lat') and fields containing at least
            the set of predictors and predictand requested for fitting the model.
        month : int
            Calendar month (January=1, Feburary=2, ...) to fit the model; if '0' then all
            months will be used
        di{lon, lat} : int
            Number of grid cells in longitude and latitude direction to include
            to compress localized synoptic modes. E.g., dilon=6, dilat=4 would use the
            local 13x9 grid cell box to compress the synoptic information at each grid cell
        predictand : str
            Name of predictand field
        predictors : list of str
            List of names of predictor fields
        {lon, lat}_range : tuple of (float, float) (optional)
            Min/max latitudes and longitudes to fit the model over (does not influence
            synoptic mode compression)
        mask : xarray.DataArray
            A dataarray with dimensions ('lon', 'lat') corresponding to 'data', indicating
            which grid cells should be excluded from fitting. Cells with "True" will be
            skipped.
        hybrid : logical
            Include SVD / compressed mode timeseries as predictors
        n_predictors : int
            Maximum number of predictors allowed in fitted model
        cross_validate : logical
            Use LOO cross-validation (one year at a time) to identify the best
            predictor variables
        verbose : logical
            Toggle progress bars and updates during modeling fitting/predicting

        """

        self._data = data
        self.verbose = verbose
        self.n_predictors = n_predictors
        self.hybrid = hybrid
        self.cv = cv

        self.grid_stack = ['lat', 'lon']
        self.cell_name = 'cell'
        self.dilon = dilon
        self.dilat = dilat

        # Fit only a subset of the data?
        self.lon_range = lon_range
        self.lat_range = lat_range
        self._fit_subset = (lon_range is not None) and (lat_range is not None)
        self.mask = mask

        self.month = month
        self.predictand = predictand
        self.predictors = predictors

        # Fitted model components
        self._grid_cell_results = []
        self._grid_cell_factors = []

        # Pre-set modeling pipelines
        # Pre_processor - Choose the month(s) we want to analyse, and then
        #                 remove the long-term trend from the data
        self.preprocessor = Pipeline([
            ('subset_time', MonthSelector(self.month)),
            ('detrend', YearlyMovingAverageDetrender())
        ])

        # Estimator - Convert a Dataset into an array, and perform a
        #             multiple linear regression
        self.estimator = Pipeline([
            ('dataset_to_array', DatasetAdapter(drop=['lat', 'lon'])),
            ('linear', LinearRegression()),
        ])

        # Fit our pre-processor for later transformation
        if self.verbose > 0:
            print("Fitting pre-processor...")
        self._data = self.preprocessor.fit_transform(self._data)
        self.to_model = self.data[self.predictors]

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


    def _fit_cell(self, ilon, ilat):
        """ For a given latitude and longitude index in a dataset, fit the regression model. """

        # if (ilon != 17) or (ilat != 4):
        #     return None, None
        #print(ilon, ilat)
        #print(self._data.isel(lon=ilon, lat=ilat))

        if self._fit_subset:
            lon = float(self._data.isel(lon=ilon).lon.values)
            lat = float(self._data.isel(lat=ilat).lat.values)

            lon_lo, lon_hi = self.lon_range
            lat_lo, lat_hi = self.lat_range

            # Check bounds and short-circuit if necessary
            if not ((lon_lo <= lon <= lon_hi) and (lat_lo <= lat <= lat_hi)):
                return None, None

        gcf = GridCellFactor(ilon, ilat, self.dilon, self.dilat)

        local_selector = DatasetSelector(sel='isel', lon=gcf.ilon, lat=gcf.ilat)
        y = local_selector.fit_transform(self.data[self.predictand])

        # Skip non CONUS datapoints. For the obs data, that'll be whenever
        # precipitation or PM2.5 is NaN
        # NOTE: This is brittle and hard-coded to our application here. It
        #       could trivially be generalized given an ilon-ilat map of
        #       a USA territory mask.
        if self.mask is not None:
            local_mask = self.mask.isel(lat=ilat, lon=ilon)
            if local_mask:
                return None, None
        elif np.all(y.isnull()):
            return None, None

        transformer = make_transformer(self.predictors, gcf, self.hybrid)
        if self.cv is None:
            _model = SelectBestFeatures(
                self.estimator, transformer, # verbose=self.verbose
            )
        else:
            _model = SelectBestFeaturesWithCV(
                self.estimator, transformer, cv=dataset_yearly_loo_cv,
                # verbose=self.verbose
            )

        # Try to fit the model
        try:
            print(gcf, end=" ")
            _model.fit(self.to_model, y)

            _score = _model.score(self.to_model, y)
            print(_score)
            gcr = GridCellResult(_model, self.predictand, _model.features_, _score)
        except:
            print("FAIL")
            gcf, gcr = None, None

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

    # def get_result_stat(self, attr, result=True):
    #     """ Fetch a result statistic from the fitted model; if `result` is False,
    #     then this will check if the attribute as attached to the GridCellResult
    #     itself instead of the OLSResult saved therein.
    #
    #     """
    #
    #     # Construct a template/strawman Dataset for holding the attribute data
    #     attr_ds = self.data.copy().drop(self.predictors)
    #     ref_field = attr_ds[self.predictand].isel(time=0)
    #     attr_ds[attr] = (ref_field.dims, np.zeros_like(ref_field.values)*np.nan)
    #     _attr_vals = attr_ds[attr].values
    #
    #     for gcr, gcf in self._gcr_gcf_iter:
    #         if gcr is None:
    #             continue
    #
    #         if result:
    #             _attr_vals[gcf.ilat, gcf.ilon] = getattr(gcr.result, attr)
    #         else:
    #             _attr_vals[gcf.ilat, gcf.ilon] = getattr(gcr, attr)
    #
    #     return attr_ds[[attr, ]]

    def predict(self, X, preprocess=True):
        """ Predict new data using the fitted models.

        """

        if preprocess:
            X = clone(self.preprocessor).fit_transform(X)

        predicted = X.copy().drop(self.predictors) * np.nan
        #_pred = predicted[self.predictand].values

        for gcr, gcf in self._gcr_gcf_iter:

            # If there was no result saved, then we should skip it (it means
            # that we probably masked out that grid cell from our analysis)
            if gcr is None:
                continue

            _y_pred = gcr.model.predict(X)
            _to_set = predicted[self.predictand].isel(lon=gcf.ilon,
                                                      lat=gcf.ilat)
            _to_set[:] = _y_pred

        return predicted

    def to_pickle(self, filename):
        """ Serialize this model to disk for later evaluation. """
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


