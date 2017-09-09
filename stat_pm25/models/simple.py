""" Collection of simplified models. """

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

from .. sklearn import (DatasetModel, MonthSelector,
                        YearlyMovingAverageDetrender, DatasetSelector,
                        FieldExtractor, Normalizer, DatasetAdapter,
                        GridCellResult)

class PCRModel(DatasetModel):
    """ Implements a principal component regression model based on our
    DatasetModel formulation.


    """

    def __init__(self, *args, month=6, n_components=3, **kwargs):
        self.month = month
        self.n_components = n_components

        # Zero out dilat and dilon, since we don't need to search around
        # neighboring grid cells
        self.dilat = self.dilon = 0

        self.preprocessor = Pipeline([
            ('subset_time', MonthSelector(self.month, width=0)),
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
            ('dataset_to_array', DatasetAdapter(drop=['lat', 'lon'])),
            ('pca', PCA(n_components=self.n_components)),
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