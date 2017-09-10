""" Collection of simplified models. """

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.base import clone

from .. sklearn import (DatasetModel, GridCellResult,
                        YearlyMovingAverageDetrender, DatasetSelector,
                        FieldExtractor, Normalizer, DatasetAdapter,
                        dataset_yearly_loo_cv)

import numpy as np

class PCRModel(DatasetModel):
    """ Implements a principal component regression model based on our
    DatasetModel formulation.


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
        """ Using the information in the passed GridCellFactor, prepare
        data transformations and fit a model.

        This method should be prepared by the user; it's the only element
        that needs to be built to create a new machine learning model. It
        should return the original GridCellFactor passed to it, as well as
        a GridCellResult encapsulating the model fit for this GridCellFactor.

        """

        # Get predictand data
        local_selector = DatasetSelector(
            sel='isel', lon=gcf.ilon, lat=gcf.ilat
        )
        y = local_selector.fit_transform(self.data[self.predictand])

        # Prepare features timelines. We want to fully include all the steps
        # to extract our features from the original, full dataset in here
        # so that our logic for re-applying the pipeline for prediction
        # later on is solvent.
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
        try:
            # print(gcf, end=" ")
            _model.fit(self.data, y)

            _score = _model.score(self.data, y)
            # print(_score)
            gcr = GridCellResult(_model, self.predictand, self.predictors, _score)
        except:
            # print("FAIL")
            gcr = None

        return gcr


class PCRModelCV(DatasetModel):
    """ Implements a principal component regression model based on our
    DatasetModel formulation.


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
        """ Using the information in the passed GridCellFactor, prepare
        data transformations and fit a model.

        This method should be prepared by the user; it's the only element
        that needs to be built to create a new machine learning model. It
        should return the original GridCellFactor passed to it, as well as
        a GridCellResult encapsulating the model fit for this GridCellFactor.

        """

        # Get predictand data
        local_selector = DatasetSelector(
            sel='isel', lon=gcf.ilon, lat=gcf.ilat
        )
        y = local_selector.fit_transform(self.data[self.predictand])

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