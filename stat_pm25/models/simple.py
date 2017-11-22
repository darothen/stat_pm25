""" Collection of simplified models. """

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.base import clone

#: local 'skleran' tools
from .. sklearn import (DatasetModel, GridCellResult,
                        YearlyMovingAverageDetrender, DatasetSelector,
                        FieldExtractor, Normalizer, DatasetAdapter,
                        dataset_yearly_loo_cv)

import numpy as np


class PCRModel(DatasetModel):
    """ Performs principal component regression model on gridded data.

    This class uses the `DatasetModel` framework to implement a principal
    component regression - a linear regression on a set of features which has
    been pre-processed using principal component analysis. It only requires a
    single additional argument on top of `DatasetModel`, the number of
    components to retain.

    This example illustrates how you can very easily put together complex
    analyses and deploy them onto your gridded model output, all using very
    simple building blocks.

    """

    def __init__(self, *args, n_components=3, **kwargs):
        # If you need to add arguments, add them as named keyword arguments in
        # the appropriate place (as shown above), and set them first in the
        # method.
        self.n_components = n_components

        # Modify any pre-set parameters, or hard-code them otherwise. For
        # instance, if you want to pre-process your data, this would be the
        # place to specify how to do so. Doing so here has the advantage that
        # you will be able to immediately apply your `predict()` method
        # to new data without pre-processing it - all that logic will be
        # saved

        # Zero out dilat and dilon, since we don't need to search around
        # neighboring grid cells
        self.dilat = self.dilon = 0

        # Set a pre-processor pipeline
        self.preprocessor = Pipeline([
            ('detrend', YearlyMovingAverageDetrender())
        ])

        # Call the parent superconstructor
        super().__init__(*args, **kwargs)

    def cell_kernel(self, gcf):
        """ Fit a model at a single grid cell.

        """

        # First, get the predictand data at the grid cell we care about. We
        # don't necessarily have to be super pedantic about this; we can just
        # use normal xarray selection methods if we want, although comments
        # below is how we could accomplish this using our specialized
        # Transformer classes
        # local_selector = DatasetSelector(
        #     sel='isel', lon=gcf.ilon, lat=gcf.ilat
        # )
        # y = local_selector.fit_transform(self.data[self.predictand])
        y = self.data[self.predictand].isel(lat=gcf.ilat, lon=gcf.ilon)

        # Prepare features timeseries. We want to fully include all the steps
        # to extract our features from the original, full dataset in here
        # so that our logic for re-applying the pipeline for prediction
        # later on will work similarly
        _model = Pipeline([
            ('subset_latlon', DatasetSelector(
                sel='isel', lon=gcf.ilon, lat=gcf.ilat)
            ),
            ('predictors', FieldExtractor(self.predictors)),
            ('normalize', Normalizer()),
            ('dataset_to_array', DatasetAdapter(drop=['lat', 'lon'])),
            ('pca', PCA(n_components=self.n_components)),
            ('linear', LinearRegression()),
        ])

        # Fit the model/pipeline
        _model.fit(self.data, y)
        # Calculate some sort of score for archival
        _score = _model.score(self.data, y)
        # Encapsulate the result within a GridCellResukt
        gcr = GridCellResult(_model, self.predictand, self.predictors, _score)

        return gcr


class PCRModelCV(DatasetModel):
    """ Similar to PCRModelCV, but incorporate cross-validation logic into
    the `cell_kernel` method.

    """

    def __init__(self, *args, n_components=3, **kwargs):
        self.n_components = n_components

        # Zero out dilat and dilon, since we don't need to search around
        # neighboring grid cells
        self.dilat = self.dilon = 0

        self.preprocessor = Pipeline([
            ('detrend', YearlyMovingAverageDetrender())
        ])

        super().__init__(*args, **kwargs)

    def cell_kernel(self, gcf):

        # Get predictand data
        y = self.data[self.predictand].isel(lat=gcf.ilat, lon=gcf.ilon)

        # Prepare features timelines
        _model = Pipeline([
            ('subset_latlon', DatasetSelector(
                sel='isel', lon=gcf.ilon, lat=gcf.ilat)
             ),
            ('predictors', FieldExtractor(self.predictors)),
            ('normalize', Normalizer()),
            ('dataset_to_array', DatasetAdapter(drop=['lat', 'lon'])),
            ('pca', PCA(n_components=self.n_components)),
            ('linear', LinearRegression()),
        ])

        # Cross-validate the prediction
        y_pred = []
        for train_inds, test_inds in dataset_yearly_loo_cv(self.data, y):
            X_train = self.data.isel(time=train_inds)
            y_train = y.isel(time=train_inds)
            X_test = self.data.isel(time=test_inds)

            _test_model = clone(_model)
            _test_model.fit(X_train, y_train)

            _yi = _test_model.predict(X_test)
            y_pred.extend(np.asarray(_yi))

        # Compute r2 score on cross-validated predictions
        _score = r2_score(y, y_pred)
        print(_score)

        # Fit the model to all of the data
        _test_model = clone(_model)
        _test_model.fit(self.data, y)
        gcr = GridCellResult(_test_model, self.predictand, self.predictors,
                             _score)

        return gcr
