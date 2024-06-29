
##################################################
##################################################
##################################################

def removeAboveThresholdPoisson(pdSeries,curSF,intervalValue=None,ZscoreThreshold=3):

    dataToReturn = pdSeries.copy() 
    if intervalValue == None:
        # If no intervalue is given, use the ZscoreThreshold value. Otherwise ZscoreThreshold is ignored.
        intervalValue = norm.cdf(ZscoreThreshold)

    dataToReturn.loc[curSF < np.log(1-intervalValue)] = np.nan 

    return dataToReturn

#TODO: Implement all functions for removing above threshold iteratively for poisson-distributions as well

def removeAboveThresholdAndRecalculatePoisson(pdSeries,curZsc,ZscoreThreshold=3,numYears=5,timeResolution='Month'):
    # Creates a copy of pdSeries in which all entries where curZsc is above ZscoreThreshold is set to NaN, and returns it together with a recalculated baseline and standard deviation
    # pdSeries should be aggregated correctly before using this function

    # curData = removeAboveThreshold(pdSeries.copy(),curZsc,ZscoreThreshold=ZscoreThreshold)
    curData = removeAboveThreshold(pdSeries,curZsc,ZscoreThreshold=ZscoreThreshold) # pdSeries gets copied inside

    curMean,curStd = rnMean(curData,numYears=numYears,timeResolution=timeResolution)

    return curData,curMean,curStd 

##################################################
##################################################
##################################################

def removeAboveThreshold(pdSeries,curZsc,ZscoreThreshold=3):
    # Returns a copy of pdSeries in which all entries where curZsc is above ZscoreThreshold is set to NaN
    # pdSeries should be aggregated correctly before using this function

    # curExc,curZsc,curExcPct = getExcessAndZscore(pdSeries,curMean,curStd)

    dataToReturn = pdSeries.copy() 
    dataToReturn.loc[curZsc[curZsc > ZscoreThreshold].index] = np.nan 

    return dataToReturn

def removeAboveThresholdAndRecalculate(pdSeries,curZsc,ZscoreThreshold=3,numYears=5,timeResolution='Month'):
    # Creates a copy of pdSeries in which all entries where curZsc is above ZscoreThreshold is set to NaN, and returns it together with a recalculated baseline and standard deviation
    # pdSeries should be aggregated correctly before using this function

    # curData = removeAboveThreshold(pdSeries.copy(),curZsc,ZscoreThreshold=ZscoreThreshold)
    curData = removeAboveThreshold(pdSeries,curZsc,ZscoreThreshold=ZscoreThreshold) # pdSeries gets copied inside

    curMean,curStd = rnMean(curData,numYears=numYears,timeResolution=timeResolution)

    return curData,curMean,curStd 

def removeAboveThresholdAndRecalculateRepeat(pdSeries,curBaseline,curStandardDeviation,ZscoreThreshold=3,numYears=5,timeResolution='Month',verbose=False):
    # Iteratively sets data outside a given ZscoreThreshold to NaN and recalculates baseline and Zscore, until all datapoints above threshold is removed.
    # pdSeries should be aggregated correctly before using this function
    # Returns raw data, final baseline and final standard deviation.

    curData = pdSeries.copy()
    curDataRemove = curData.copy()

    numAboveThreshold = 1
    # Count number of iterations (for printing)
    numIter = 0

    while numAboveThreshold > 0:
        # Determine number of entries above threshold
        _,curZsc,_ = getExcessAndZscore(curDataRemove,curBaseline,curStandardDeviation)
        numAboveThreshold = (curZsc > ZscoreThreshold).sum()

        if verbose:
            # Increment counter
            numIter += 1
            print(f'Iteration {numIter} of removing larger crises. {numAboveThreshold} found.')
            # print(f'Count above threshold: {numAboveThreshold}')


        curDataRemove,curBaseline,curStandardDeviation = removeAboveThresholdAndRecalculate(curDataRemove,curZsc,ZscoreThreshold=ZscoreThreshold,numYears=numYears,timeResolution=timeResolution)

    # # Once everything has been removed, recalculate mean and std with original data
    # curBaseline,curStandardDeviation = rnMean(curDataRemove,numYears=numYears,timeResolution=timeResolution)


    return curData,curBaseline,curStandardDeviation 

def removeAboveThresholdAndRecalculateRepeatFull(pdSeries,ZscoreThreshold=3,numYears=5,timeResolution='Month',verbose=False):
    # Calculates mean and standard deviation and runs the removeAboveThresholdAndRecalculateRepeat function (see above)
    # pdSeries should be aggregated correctly before using this function
    # Returns raw data, final baseline and final standard deviation.

    curBaseline,curStandardDeviation = rnMean(pdSeries,numYears=numYears,timeResolution=timeResolution,distributionType='Standard')

    curData,curBaseline,curStandardDeviation  = removeAboveThresholdAndRecalculateRepeat(pdSeries,curBaseline,curStandardDeviation,ZscoreThreshold=ZscoreThreshold,numYears=numYears,timeResolution=timeResolution,verbose=verbose)

    return curData,curBaseline,curStandardDeviation 


def runFullAnalysisDailySeries(pdSeries,numYears = 12,ZscoreThreshold=3,verbose=False):
    # Assumes pdSeries has datetime64 as index 
    # Note that if data has to be averaged by week (e.g. because sundays are more common as burial days than any other weekday), this should be done *before* running this function.

    # Make a copy, to avoid overwriting things
    pdSeries = pdSeries.copy()

    # Run analysis of all data
    _,curBaseline,curStandardDeviation = removeAboveThresholdAndRecalculateRepeatFull(pdSeries,ZscoreThreshold=ZscoreThreshold,numYears=numYears,timeResolution='Day',verbose=verbose)

    # Also calculate the residuals with the corrected baseline
    curExcess = pdSeries - curBaseline 
    curZscore = curExcess/curStandardDeviation  
    curExcessPct = 100 * curExcess/curBaseline

    # Return everything
    return curBaseline,curStandardDeviation,curExcess,curZscore,curExcessPct