#!/usr/bin/env python
""" Fit the Shen et al (2017) model for a specific dataset. """
from stat_pm25.models import SimpleRegress, OldShen2017Model

import numpy as np
import xarray as xr

from argparse import ArgumentParser, RawDescriptionHelpFormatter
parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("dataset",
                    help="The gridded dataset to use for fitting the model")
parser.add_argument("output",
                    help="Name of output file containing serialized, fitted model")
parser.add_argument("--month", required=True, type=int,
                    help="Month (1-12) of year to fit model to")
parser.add_argument("--case", default="local", choices=['local', 'hybrid'], type=str,
                    help="Specifier for which parameter set to include as predictands")
parser.add_argument("--cv", action='store_true',
                    help="Enable cross-validation on model fitting")


if __name__ == "__main__":

    args = parser.parse_args()
    
    predictand = 'PM25'
    predictors = ['TEMP', 'RH', 'PRECIP', 'U', 'V']
    dilon, dilat = 6, 4
    
    obs_data = xr.open_dataset(args.dataset)
    # Mask out cells where we have no PRECIP data
    mask = np.isnan(obs_data.PRECIP.isel(time=0)).rename("CONUS_MASK")
    
    do_hybrid = args.case == 'hybrid'
    print("Initializing model...")
    # obs_model = SimpleRegress(
    #     obs_data, predictand, predictors, month=args.month, mask=mask,
    #     # lat_range=(30, 33), lon_range=(-80, -78),
    #     verbose=True, # n_predictors=3, hybrid=do_hybrid, cv=args.cv
    # )
    obs_model = OldShen2017Model(
        obs_data,
        month=args.month, dilon=dilon, dilat=dilat,
        predictand=predictand,
        predictors=predictors, mask=mask,
        hybrid=do_hybrid, n_predictors=3, cv=args.cv,
        verbose=True
    )

    print("Fitting model...")
    # Fit the model
    obs_model.fit_parallel(-1)
    
    # Save output
    print("Saving to", args.output)
    obs_model.to_pickle(args.output)

    # Test prediction
    print("Making test prediction")
    obs_model.predict(obs_data).to_netcdf("test.pred.nc")
