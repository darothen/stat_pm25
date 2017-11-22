import xarray as xr
import numpy as np

from sklearn.pipeline import  Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

from stat_pm25.sklearn import (GridCellFactor, DatasetSelector, MonthSelector,
                               YearlyMovingAverageDetrender, FieldExtractor,
                               Normalizer, Stacker, DatasetAdapter,
                               DatasetFeatureUnion, dataset_yearly_loo_cv)
from stat_pm25.models.shen2017 import (SelectBestFeatures,
                                       SelectBestFeaturesWithCV,
                                       SVDModeCompressor)


grid_stack = ['lat', 'lon']
cell_name = 'cell'
predictand = 'PM25'
predictors = ['TEMP', 'RH', 'PRECIP', 'U', 'V']
dilon, dilat = 6, 4
month = 6

data = xr.open_dataset("data/obs.usa_subset.nc")
mask = np.isnan(data.PRECIP.isel(time=0)).rename("CONUS_MASK")

# site_lon = -82.5
# site_lat = 32.5
# ilon = find_nearest(data.lon.values, site_lon)
# ilat = find_nearest(data.lat.values, site_lat)
ilon, ilat = 18, 8
# ilon, ilat = 17, 6
# ilon, ilat = 18, 2
print(ilon, ilat)
gcf = GridCellFactor(ilon, ilat, dilon=dilon, dilat=dilat)

local_selector = DatasetSelector(sel='isel', lon=gcf.ilon, lat=gcf.ilat)
area_selector = DatasetSelector(sel='isel',
                                lon=gcf.ilon_range, lat=gcf.ilat_range)

pre_process = Pipeline([
    ('subset_time', MonthSelector(month)),
    ('detrend', YearlyMovingAverageDetrender())
])

local_pipeline = Pipeline([
    ('predictors', FieldExtractor(predictors)),
    ('normalize', Normalizer()),
    ('subset_latlon', local_selector),
])

hybrid_pipeline = Pipeline([
    ('predictors', FieldExtractor(predictors)),
    ('subset_latlon', area_selector),
    ('normalize', Normalizer()),
    ('stack', Stacker(['lon', 'lat'], 'cell')),
    ('svd_modes', SVDModeCompressor(predictors, max_modes=2)),
])

estimation_pipeline = Pipeline([
    ('dataset_to_array', DatasetAdapter(drop=['lat', 'lon'])),
    ('linear', LinearRegression()),
])

fused_estimator = Pipeline([
    ('features', DatasetFeatureUnion([
        ('local', local_pipeline),
        ('hybrid', hybrid_pipeline),
    ])),
    ('model', estimation_pipeline),
])

to_model = pre_process.fit_transform(data)
X = to_model[predictors]
y = local_selector.fit_transform(to_model[predictand])

## Straightforward estimation with all 10 features
print("Regression on all features")
fused_estimator.fit(to_model, y)
print(fused_estimator.score(to_model, y))
print("\n", "--"*40)

## Cross-validated best feature selection
print("Cross-validated feature selection")
estimator = estimation_pipeline
transformer = Pipeline([
    ('features', DatasetFeatureUnion([
        ('local', local_pipeline),
        # ('pca', PCA())
        # ('hybrid', hybrid_pipeline)
    ])),
])
cv = dataset_yearly_loo_cv
n_features = 3
verbose = 1

X = transformer.fit_transform(to_model, y)
# BCF = SelectBestFeaturesWithCV(estimator, transformer, cv, verbose=verbose)
BCF = SelectBestFeatures(estimator, transformer, verbose=verbose)
BCF.fit(to_model, y)

pred = BCF.predict(to_model)
from scipy.stats import pearsonr
print(pearsonr(y, pred)[0]**2)

import seaborn as sns
jg = sns.jointplot(y, pred, size=4, color='k')
xlo, xhi = jg.ax_joint.get_xlim()
ylo, yhi = jg.ax_joint.get_ylim()
lo, hi = np.min([xlo, ylo]), np.max([xhi, yhi])
jg.ax_joint.plot([lo, hi], [lo, hi], color='k', lw=1., ls='dotted')
jg.ax_joint.set_xlim(lo, hi)
jg.ax_joint.set_xlabel("Observed")
jg.ax_joint.set_ylim(lo, hi)
jg.ax_joint.set_ylabel("Statistical Model")


##########

pcr_pipeline = Pipeline([
    ('local', local_pipeline),
    ('ds_to_ar', DatasetAdapter(drop=['lat', 'lon'])),
    ('pca', PCA()),
    ('linear', LinearRegression()),
])
pcr_pipeline.fit(to_model, y)
yp = pcr_pipeline.predict(to_model)
jg = sns.jointplot(y, pred, size=4, color='k')
xlo, xhi = jg.ax_joint.get_xlim()
ylo, yhi = jg.ax_joint.get_ylim()
lo, hi = np.min([xlo, ylo]), np.max([xhi, yhi])
jg.ax_joint.plot([lo, hi], [lo, hi], color='k', lw=1., ls='dotted')
jg.ax_joint.set_xlim(lo, hi)
jg.ax_joint.set_xlabel("Observed")
jg.ax_joint.set_ylim(lo, hi)
jg.ax_joint.set_ylabel("Statistical Model")