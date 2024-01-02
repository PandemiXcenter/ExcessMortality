# Excess mortality functions

## Collection of functions for the calculations of excess mortality for use in the PandemiX research group

This repository contains python code for calculating excess mortality from historical mortality data.

The functions defined here assume data is available both before and after events in questions. Baseline calculations are extended to both ends of the available data, but trends in data will not be sufficiently captured at the ends. For this reason, the functions here are mostly relevant for historical data, and _not_ for modern data. 
For modern data, other methods should be used for calculating excess mortality while accounting for e.g. demographic trends and age-distribution. 

- Rasmus Kristoffer Pedersen, 2024