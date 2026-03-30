Detect precuror events from Type II supernovae.

## Setup

Make sure your virtual environment is running. 

Install the requirements. 

```bash
pip install -r requirements.txt
```

## Files

*HGP.ipynb* - classification using Heteroscedastic GP. I use matern kernel instead of RBF due to inherent roughness of the light curve data (RBF smooths everything out)

*HGP-Anomalies.ipynb* - wip. Fit an HGP to anomalies only, check if input light curves match the GP prediction (this means they are close to anomalies and should be classified as such).

*DBSCAN.ipynb* - deprecated. DBSCAN is a clustering algorithm. The idea was to normalize the curves, create segments of each curve and classify them using the clustering algorithm. If assigned to cluster -1, then the curve is an anomaly.

*upload.ipynb* - spam the Alerce API for new light curves.

*AntiAGN-inator.ipynb* - get rid of AGN light curves by checking if nearby objects include a galaxy center.

*AntiTransient-inator.ipynb* - get rid of transient data by using an FFT. Not very consistent, might fit an HGP using the periodic kernel, split into training and test data, fit HGP to training data, then check if test loss is very small (this means the data is periodic).

*HGP.py* - contains the logic for HGP.

*load.py* - loads the csv_files in the _csv_files_ dir. These csv files were generated using _upload.py_
