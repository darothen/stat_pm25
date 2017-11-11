# Statistical PM2.5 Modeling

This is a working project folder containing experiments and software engineering geared towards studying relationships between meteorology and air quality. It began as an independent validation of the study by [Shen et al, 2017][shen_2017], but evolved into a more complex study aimed at fundamental questions about statistical modeling of these relationships and their utility with respect to detailed chemical transport modeling.

I've published this work-in-progress since I have stepped away from full-time basic research. Some work is deliberately left out of this repository (namely, the code and data from Shen's work, as well as the data from the ensemble simulations I studied here; it's still available on request, but I have not yet received permission to publish it and it is not licensed in such a way I feel comfortable doing so without said permission). However, I mostly want to highlight the work accomplished here with regards to building flexible spatio-temporal models using idioms and tools within the PyData community and software toolkit. Those are detailed below.

## Installation / Dependencies

You should use the attached [environment.yml] to setup an environment for running this package; it should simplify the process of getting the necessary dependencies. First, execute:

    $ conda env create -f environment.yml

This will create a virtual environment called *stat_pm25*, which will contain all the dependencies to use this package and reproduce this work. To use this code, first activate the environment

    $ source activate stat_pm25

and then use **setuptools** to install this package:

    $ pip install -e .

You should then be able to import everything and work with it.

## Introduction to Modeling Tools

The premise of this package is to extend [`Pipeline`](http://scikit-learn.org/stable/modules/pipeline.html) idiom from scikit-learn to work with modeling tasks using gridded geophysical data. The fundamental problem we aim to solve is one of code simplicity and modularization. Fundamentally, we have two different pieces of information at our disposal when we perform statistical modeling on geophysical data: local timeseries of variables, and gridded fields of data. Building robust machine learning or statistical estimators on this data becomes challenging, because we have to support these two very different types of information.

For instance, many statistical modeling applications in the atmospheric chemistry domain involve fitting timeseries models at each grid cell in ones dataset. This is an embarrassingly parallel problem, but it doesn't map to standard modeling idioms (repeating a model over many sets of data, and then predicting new fields of data). We can instead decompose our problem to fit a model to each grid cell, but there's a significant amount of coding overhead associated with this, and it can be hard to optimize this code for performance. But a bigger problem emerges when we wish to use spatio-temporal information in our models, which breaks the independence assumption we shoe-horn into our code when we attack it from a grid-cell-by-grid-cell perspective.

### The `DatasetModel`

This toolkit aims to solve some of these problems by implementing a `DatasetModel` object. A `DatasetModel` consumes an [xarray `Dataset`](http://xarray.pydata.org/en/stable/data-structures.html#dataset) containing all of the raw data used in a modeling task, including the predictors and predictand data. It includes logic to first use a `preprocessor` constructed as a scikit-learn `Pipeline` to munge the raw data into a suitable form for modeling, and then apply a `cell_kernel()` method to each grid-cell in the data to fit a statistical model.

A `DatasetModel` is really an *abstract base class*. To use it, one should instantiate a specialized object which inherits from `DatasetModel`.

### Simple Example

Suppose you have a dataset with dimensions (time, lon, lat), with 3 predictor fields *x*, *y*, and *z*, and a a predictand field *p*. You wish to model the linear relationship between these three predictors and *p* at each point in your dataset, using the timeseries at each grid cell. To do this, you might first implement the following extension to `DatasetModel`:

``` python
class LinearModel(DatasetModel):

    def __init__(self, *args, **kwargs):

        # Over-ride the parameters for searching neighborhoods around cells,
        # since we don't need it in this example
        self.dilat = self.dilon = 0

        # Don't define a pre-processor - assume we don't need one
        self.preprocessor = None

        # Call the parent constructor
        super().__init__(*args, **kwargs)

```

We also need to write a `cell_kernel()` method for `LinearModel`, so that it knows how to fit a model at each grid cell:

``` python
    def cell_kernel(self, gcf):

        # Select the local predictand data
        local_selector = DatasetSelector(
            sel='isel', lon=gcf.ilon, lat=gcf.ilat
        )
        y = local_selector.fit_transform(self.data[self.predictand])
        # y = self.data[self.predictand].isel(lat=gcf.ilat, lon=gcf.ilon)

        # Build a pipeline to fit
        _model = Pipeline([
            ('subset_latlon', DatasetSelector(
                sel='isel', lon=gcf.ilon, lat=gcf.ilat)
            ),
            ('predictors', FieldExtractor(self.predictors)),
            ('normalize', Normalizer()),
            ('dataset_to_array', DatasetAdapter(drop=['lat', 'lon'])),
            ('linear', LinearRegression()),
        ])

        # Fit the model/pipeline
        _model.fit(self.data, y)
        # Calculate some sort of score for archival
        _score = _model.score(self.data, y)
        # Encapsulate the result within a GridCellResukt
        gcr = GridCellResult(_model, self.predictand, self.predictors, _score)

        return gcr
```

This will seem complicated at first, but we can break it down. First, note that `cell_kernel` takes two arguments - *self*, since it's a class method, and *gcf*. A *gcf* is a `GridCellFactor`, which is just a thin wrapper containing the information on how a model should use local and neighboring data in your `DataSet`. A `GridCellFactor` will always contain the instance variables *ilat* and *ilon*, which let you target which cell you're looking at in a `Dataset`. It also has some hooks to help you select a range of cells, if for instance your method needed to look at a neighborhood around a function - in fact, these are the *dilon* and *dilat* class values we over-rode in the constructor for `LinearModel`.

The rest of the work we encode is exactly what you'd independently write if you were performing a traditional scikit-learn analysis. First, we grab the local timeseries for our predictand:

``` python
        local_selector = DatasetSelector(
            sel='isel', lon=gcf.ilon, lat=gcf.ilat
        )
        y = local_selector.fit_transform(self.data[self.predictand])
```

We do this with the help of a `DatasetSelector`, which is a specialized type of [scikit-learn transformer](http://scikit-learn.org/stable/data_transforms.html) which consumes and emits a `Dataset`. We then fit and apply the transformer to the data we read in as part of a `DatasetModel`.

Next, we build the actual model pipeline:

``` python
        _model = Pipeline([
            ('subset_latlon', DatasetSelector(
                sel='isel', lon=gcf.ilon, lat=gcf.ilat)
            ),
            ('predictors', FieldExtractor(self.predictors)),
            ('normalize', Normalizer()),
            ('dataset_to_array', DatasetAdapter(drop=['lat', 'lon'])),
            ('linear', LinearRegression()),
        ])
```

This is just a sequence of tasks which help prepare/transform our data, then perform an analysis:

1. First a `DatasetSelector` like before, to extract just the local timeseries of data
2. Use a `FieldExtractor` to subset the predictors
3. De-mean and standardize the data using `Normalizer` (which will cache these parameters for future use, too)
4. Use a `DatasetAdapter` to convert from a `Dataset` to the type of NumPy array that scikit-learn likes to use; this is *always* included in any pipeline you build and should be the last step before running the actual models
5. Fit a `LinearRegression` on our prepared data

None of these tasks is complex, but writing them as a pipeline helps clarify the flow of data in your analysis task, as well as simplifies the logic necessary later on for applying cross-validation or other advanced analyses.

With the pipeline encoded, fitting the model to your data is a trivial one-liner:

``` python
        # Fit the model/pipeline
        _model.fit(self.data, y)
```

Note that aside from selecting the local predictand timeseries, *y*, we haven't had to actually do anything to our data. That's the beauty of this idiom - we abstract away all the logic for preparing our data, and apply it universally.  The only remaining task is to save your results and return them, which we do with the help of another wrapper class, a `GridCellResult`:

``` python
        # Calculate some sort of score for archival
        _score = _model.score(self.data, y)
        # Encapsulate the result within a GridCellResukt
        gcr = GridCellResult(_model, self.predictand, self.predictors, _score)

        return gcr
```

A `GridCellResult` caches the model that you fit, any input parameters to it, and any additional information you might want to store. It's standard to compute a model *score* and save it as part of a `GridCellResult`. But a `GridCellResult` is really just a fancy dictionary, and you can add any keyword arguments you want.

The `cell_kernel()` returns the `GridCellResult`, which is kept track of internally by `DatasetModel`. The `DatasetModel` has all the utilities you need to then compute new predictions using new data, as well as to grab any saved score information from your fitting process. It'll always try to return a `Dataset` with the same grid and dimensions as your input data, so that downstream visualization and analysis is that much easier.

## Example Modeling Tools

I've implemented two models for reference using this framework inside the *models/* directory: an implementation of the [Shen et al, 2017][shen_2017] model, and a simple principal component regression with tunable *n_components.* For each model, I've also included an extended implementation with time-of-fit cross-validation, to show how you can augment your models with these custom cross-validators.

The file *sklearn.py* contains the core modeling system implementation, as well as a bunch of reference transformer objects:

- `DatasetFeatureUnion` - sequentially apply lists of transformers and concatenate into a new `Dataset`
- `FieldExtractors` - subset fields from a `Dataset`
- `MonthSelector` - select timesteps within a `Dataset` surrounding a given month
- `DatasetSelector` - apply dimensional indexing/selecting to a `Dataset`
- `Normalizer` - center and scale each field in a `Dataset`
- `Stacker` - transform a `Dataset` by stacking the given dimensions into one multi-indexed 'super' dimension
- `YearlyMovingAverageDetrender` - group a `Dataset` by month and remove the local moving average for that month across all years of data
- `DatasetFunctoinTransformer` - apply an arbitrary function to each field in a `Dataset`
- `DatasetAdapter` - convert a multi-field `Dataset` into a 2D array such that each field is contained in a separate column


[shen_2107]: www.atmos-chem-phys.net/17/4355/2017/
