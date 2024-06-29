
def removeAboveThreshold(pdSeries,curUncertainty,ZscoreThreshold=3,intervalValue=None,distributionType='Standard'):

    # Make a copy, to avoid overwriting
    dataToReturn = pdSeries.copy() 

    if distributionType == 'Standard':
        # If distribution type is Standard, the curUncertainty should be the zscore.
        dataToReturn.loc[curUncertainty[curUncertainty > ZscoreThreshold].index] = np.nan 

    elif distributionType == 'Poisson':
        # If distribution type is Standard, the curUncertainty should be logsf.

        if intervalValue == None:
            # If no intervalue is given, use the ZscoreThreshold value. Otherwise ZscoreThreshold is ignored.
            intervalValue = norm.cdf(ZscoreThreshold)

        dataToReturn.loc[curUncertainty < np.log(1-intervalValue)] = np.nan 

        
    return dataToReturn


def removeAboveThresholdAndRecalculate(pdSeries,curUncertainty,ZscoreThreshold=3,numYears=5,timeResolution='Month',intervalValue=None,distributionType='Standard'):
    # Creates a copy of pdSeries in which all entries where curUncertainty is above ZscoreThreshold is set to NaN, and returns it together with a recalculated baseline and the new uncertainty measure
    # pdSeries should be aggregated correctly before using this function

    curData = removeAboveThreshold(pdSeries,curUncertainty,ZscoreThreshold=ZscoreThreshold,intervalValue=intervalValue,distributionType=distributionType) # pdSeries gets copied inside

    curMean,curUncertainty = rnMean(curData,numYears=numYears,timeResolution=timeResolution,distributionType=distributionType)

    return curData,curMean,curUncertainty 


def removeAboveThresholdAndRecalculateRepeat(pdSeries,curBaseline,curUncertainty,ZscoreThreshold=3,numYears=5,timeResolution='Month',verbose=False,intervalValue=None,distributionType='Standard'):
    # Iteratively sets data outside a given ZscoreThreshold to NaN and recalculates baseline and Zscore, until all datapoints above threshold is removed.
    # pdSeries should be aggregated correctly before using this function
    # Returns raw data, final baseline and final standard deviation.

    curData = pdSeries.copy()
    curDataRemove = curData.copy()

    numAboveThreshold = 1
    # Count number of iterations (for printing)
    numIter = 0

    while numAboveThreshold > 0:

        if distributionType == 'Standard':
                
            # Determine number of entries above threshold
            _,curZsc,_ = getExcessAndZscore(curDataRemove,curBaseline,curUncertainty)
            numAboveThreshold = (curZsc > ZscoreThreshold).sum()

            if verbose:
                # Increment counter
                numIter += 1
                print(f'Iteration {numIter} of removing larger crises. {numAboveThreshold} found.')
                # print(f'Count above threshold: {numAboveThreshold}')


        curDataRemove,curBaseline,curUncertainty = removeAboveThresholdAndRecalculate(curDataRemove,curZsc,ZscoreThreshold=ZscoreThreshold,numYears=numYears,timeResolution=timeResolution,intervalValue=intervalValue,distributionType=distributionType)

    # # Once everything has been removed, recalculate mean and std with original data
    # curBaseline,curUncertainty = rnMean(curDataRemove,numYears=numYears,timeResolution=timeResolution)


    return curData,curBaseline,curUncertainty 


# def removeAboveThresholdPoisson(pdSeries,curSF,intervalValue=None,ZscoreThreshold=3):

#     if intervalValue == None:
#         # If no intervalue is given, use the ZscoreThreshold value. Otherwise ZscoreThreshold is ignored.
#         intervalValue = norm.cdf(ZscoreThreshold)

#     dataToReturn.loc[curSF < np.log(1-intervalValue)] = np.nan 

#     return dataToReturn

# #TODO: Implement all functions for removing above threshold iteratively for poisson-distributions as well



# def removeAboveThreshold(pdSeries,curZsc,ZscoreThreshold=3):
#     # Returns a copy of pdSeries in which all entries where curZsc is above ZscoreThreshold is set to NaN
#     # pdSeries should be aggregated correctly before using this function

#     # curExc,curZsc,curExcPct = getExcessAndZscore(pdSeries,curMean,curStd)

#     dataToReturn = pdSeries.copy() 
#     dataToReturn.loc[curZsc[curZsc > ZscoreThreshold].index] = np.nan 

#     return dataToReturn

# def removeAboveThresholdAndRecalculate(pdSeries,curZsc,ZscoreThreshold=3,numYears=5,timeResolution='Month'):
#     # Creates a copy of pdSeries in which all entries where curZsc is above ZscoreThreshold is set to NaN, and returns it together with a recalculated baseline and standard deviation
#     # pdSeries should be aggregated correctly before using this function

#     # curData = removeAboveThreshold(pdSeries.copy(),curZsc,ZscoreThreshold=ZscoreThreshold)
#     curData = removeAboveThreshold(pdSeries,curZsc,ZscoreThreshold=ZscoreThreshold) # pdSeries gets copied inside

#     curMean,curStd = rnMean(curData,numYears=numYears,timeResolution=timeResolution)

#     return curData,curMean,curStd 

# def removeAboveThresholdAndRecalculateRepeat(pdSeries,curBaseline,curStandardDeviation,ZscoreThreshold=3,numYears=5,timeResolution='Month',verbose=False):
#     # Iteratively sets data outside a given ZscoreThreshold to NaN and recalculates baseline and Zscore, until all datapoints above threshold is removed.
#     # pdSeries should be aggregated correctly before using this function
#     # Returns raw data, final baseline and final standard deviation.

#     curData = pdSeries.copy()
#     curDataRemove = curData.copy()

#     numAboveThreshold = 1
#     # Count number of iterations (for printing)
#     numIter = 0

#     while numAboveThreshold > 0:
#         # Determine number of entries above threshold
#         _,curZsc,_ = getExcessAndZscore(curDataRemove,curBaseline,curStandardDeviation)
#         numAboveThreshold = (curZsc > ZscoreThreshold).sum()

#         if verbose:
#             # Increment counter
#             numIter += 1
#             print(f'Iteration {numIter} of removing larger crises. {numAboveThreshold} found.')
#             # print(f'Count above threshold: {numAboveThreshold}')


#         curDataRemove,curBaseline,curStandardDeviation = removeAboveThresholdAndRecalculate(curDataRemove,curZsc,ZscoreThreshold=ZscoreThreshold,numYears=numYears,timeResolution=timeResolution)

#     # # Once everything has been removed, recalculate mean and std with original data
#     # curBaseline,curStandardDeviation = rnMean(curDataRemove,numYears=numYears,timeResolution=timeResolution)


#     return curData,curBaseline,curStandardDeviation 
