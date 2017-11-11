#!/usr/bin/env python
""" Fit the Shen et al (2017) model for a specific dataset. """
from stat_pm25.models.simple import PCRModel, PCRModelCV

import numpy as np
import xarray as xr

from argparse import ArgumentParser, RawDescriptionHelpFormatter
parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("dataset",
                    help="The gridded dataset to use for fitting the model")
parser.add_argument("output",
                    help="Name of output file containing serialized, fitted model")


if __name__ == "__main__":

    args = parser.parse_args()
    
    predictand = 'PM25'
    predictors = ['TEMP', 'RH', 'PRECIP', 'U', 'V']
    
    obs_data = xr.open_dataset(args.dataset)
    # Mask out cells where we have no PRECIP data
    mask = np.isnan(obs_data.PRECIP.isel(time=0)).rename("CONUS_MASK")
    
    print("Initializing model...")
    obs_model = PCRModel(
        obs_data, predictand, predictors, n_components=3, mask=mask,
        verbose=True,
    )

    print("Fitting model...")
    # Fit the model
    obs_model.fit_parallel(-1)
    
    # Save output
    print("Saving to", args.output)
    obs_model.to_pickle(args.output)

    # Test prediction
    print("Making test prediction")
    y = obs_model.data[[predictand, ]].rename(
        {predictand: predictand+'_ref'}
    )
    y_hat = obs_model.predict(obs_data)
    score = obs_model.score.to_dataset()
    result = xr.auto_combine([y, y_hat, score])
    result.to_netcdf("linear.pred.nc")
