# --- Import dependencies ---
import numpy as np 
import pandas as pd 
import math 
from scipy.stats import poisson,norm



# Simple functions for running mean, using convolution. Input should be a numpy array
def rnMeanSimple(data,meanWidth=7):
    return np.convolve(data, np.ones(meanWidth)/meanWidth, mode='valid')
def rnTimeSimple(t,meanWidth=7):
    return t[math.floor(meanWidth/2):-math.ceil(meanWidth/2)+1]


def groupByYear(pdSeries):
    # Groups data by year
    # 'pdSeries' should be a pandas series of data, with a datetime-index. 
    # Returns the series aggregated by year.
    pdSeries = pdSeries.groupby([pdSeries.index.year.rename('Year')]).sum().reset_index()

    pdSeries['Date'] = pd.to_datetime(dict(year=pdSeries.Year,month=np.ones(len(pdSeries.Year)),day=np.ones(len(pdSeries.Year))))

    pdSeries = pdSeries.drop(columns='Year').set_index('Date').iloc[:,0]
    
    return pdSeries

def groupByMonth(pdSeries):
    # Groups data by month
    # 'pdSeries' should be a pandas series of data, with a datetime-index. 
    # Returns the series aggregated by year and month.
    pdSeries = pdSeries.groupby([pdSeries.index.year.rename('Year'),pdSeries.index.month.rename('Month')]).sum().reset_index()

    pdSeries['Date'] = pd.to_datetime(dict(year=pdSeries.Year,month=pdSeries.Month,day=np.ones(len(pdSeries.Year))))

    pdSeries = pdSeries.drop(columns=['Year','Month']).set_index('Date').iloc[:,0]
    
    return pdSeries
    
def groupByWeek(pdSeries):
    # Groups data by week
    # 'pdSeries' should be a pandas series of data, with a datetime-index. 
    # Returns the series aggregated by year and isocalendar week.
    pdSeries = pdSeries.groupby([pdSeries.index.isocalendar().year,pdSeries.index.isocalendar().week]).sum().reset_index()


    pdSeries['Date'] = pdSeries.apply(lambda x: pd.Timestamp.fromisocalendar(int(x.year),int(x.week),1),axis=1)
    pdSeries = pdSeries.sort_values('Date').set_index('Date').drop(columns=['year','week']).iloc[:,0]

    return pdSeries

def reshapePivot(pivotTable,timeResolution='Month'):
    # Reshapes pivottables into series

    if timeResolution == 'Year':    
        pivotTable = pivotTable.reset_index()

        pivotTable['Date'] = pd.to_datetime(dict(
            year=pivotTable.Year,
            month=np.ones(len(pivotTable.Year)),
            day=np.ones(len(pivotTable.Year))))

        pivotTable = pivotTable.sort_values('Date').set_index('Date').drop(columns=['Year']).iloc[:,0]

    elif timeResolution == 'Month':
        pivotTable = pivotTable.reset_index().melt(id_vars='Year')

        pivotTable['Date'] = pd.to_datetime(dict(
            year=pivotTable.Year,
            month=pivotTable.Month,
            day=np.ones(len(pivotTable.Year))))

        pivotTable = pivotTable.sort_values('Date').set_index('Date').drop(columns=['Year','Month']).iloc[:,0]
        
    elif timeResolution == 'Week':
        # Determine which years have week 53 (to remove those without)
        firstDate = np.datetime64(pivotTable.index[0].astype(str)+'-01-01') # First date in first year with data
        lastDate = np.datetime64((pivotTable.index[-1]+1).astype(str)+'-01-01') # First date in the next year after end of data
        allIsoDF = pd.Series(np.arange(firstDate,lastDate,np.timedelta64(1,'D'))).dt.isocalendar() # Get a range of dates from the earliest possible to the last possible
        YearsWith53 = allIsoDF[allIsoDF.week==53].year.unique() # Determine which years have a week 53

        pivotTable = pivotTable.reset_index().melt(id_vars='year') # Melt pivottable
                
        # Drop week 53 in years that did not have a week 53
        pivotTable = pivotTable.drop(pivotTable[(~pivotTable.year.isin(YearsWith53)) & (pivotTable.week == 53)].index)

        # Determine date from week and year
        pivotTable['Date'] = pivotTable.apply(lambda x: pd.Timestamp.fromisocalendar(x.year,x.week,1),axis=1)

        # Drop extra columns and return sorted series
        pivotTable = pivotTable.sort_values('Date').set_index('Date').drop(columns=['year','week']).iloc[:,0]

    elif timeResolution == 'Day':
        
        pivotTable = pivotTable.reset_index().melt(id_vars='Year') # Melt pivottable

        pivotTable['Date'] = pd.to_datetime(dict(
            year=pivotTable.Year,
            month=pivotTable.Month,
            day=pivotTable.Day),errors='coerce') # Make a date-columns (coerce "false" leap-days to NaT, i.e. in non-leap years)
            
        pivotTable = pivotTable.sort_values('Date').set_index('Date').drop(columns=['Year','Month','Day']).iloc[:,0] # Sort by date and drop extra columns

        pivotTable = pivotTable.loc[pivotTable.index.notna()] # Remove invalid dates (leap-days in not leap-years)


    return pivotTable


def removeLeapDays(pdSeries):
    # Removes leap days from pandas series
    
    # Get indices of leap days
    curDates = pdSeries.index
    nonLeapDay = ~((curDates.month == 2) & (curDates.day == 29))

    return pdSeries.loc[nonLeapDay]

def seriesToPivot(pdSeries,timeResolution='Month'):
    # Helper function for restructuring a pandas series into a pivot-table for rolling-calculations
    if timeResolution == 'Year':

        # Start by grouping data by year, in case it's daily or monthly resolution
        # (the min_count flag makes sure that if all are NaN, NaN is actually used instead of 0)
        serYear = pdSeries.groupby(pdSeries.index.year.rename('Year')).sum(min_count=1)
        curPivot = serYear 

    elif timeResolution == 'Month':
        # Start by grouping data by month, in case it's on daily resolution
        serMonth = pdSeries.groupby([pdSeries.index.year.rename('Year'),pdSeries.index.month.rename('Month')]).sum(min_count=1).to_frame() 

        # Organize as pivot table
        curPivot = serMonth.pivot_table(serMonth.columns[0],index='Year',columns='Month')

    elif timeResolution == 'Week':
        
        # Group by week (using isocalendar weeks and isocalendar years)
        serWeek = pdSeries.groupby([pdSeries.index.isocalendar().year,pdSeries.index.isocalendar().week]).sum(min_count=1).to_frame() 

        # Organize as pivot table
        curPivot = serWeek.pivot_table(values=serWeek.columns[0],index='year',columns='week')

    elif timeResolution == 'Day':
        # # Ignore leap day 
        # pdSeries = removeLeapDays(pdSeries)
        # # In fact, removing leapdays is no longer necessary with the current method. Using rolling, the leap days will simply give a value of NaN, and the baseline will be blank on leap days

        # Add columns for year, month and day
        curFrame = pdSeries.to_frame()
        curFrame['Year'] = curFrame.index.year 
        curFrame['Month'] = curFrame.index.month
        curFrame['Day'] = curFrame.index.day
        
        # Organize as pivot-table (with multi-columns)
        # curPivot = curFrame.pivot_table(values=pdSeries.name,index='Year',columns=['Month','Day'])
        # curPivot = curFrame.pivot_table(values=curFrame.columns[0],index='Year',columns=['Month','Day'])
        curPivot = curFrame.pivot_table(values=curFrame.columns[0],index='Year',columns=['Month','Day'],dropna=False)


        ############ 
        # Dirty temporary hack for the rare situation where all dates in an entire year gets counted as outliers
        # (This happens when analyzing data on a daily basis for Holbæk county and assuming poisson-distributed noise)
        ############ 
        yearMin = np.min(curPivot.index)
        yearMax = np.max(curPivot.index)
        yearRange = np.arange(yearMin,yearMax+1)
        for y in yearRange:
            if y not in curPivot.index:
                curPivot.loc[y] = np.nan 
        curPivot = curPivot.sort_index()

    return curPivot

# Function for calculating running mean from surrounding data
# Assumes input is pandas series, and uses pandas 'rolling' to determine mean
# Index should be a datetimeindex, with correct dates
def rnMean(pdSeries,numYears=5,timeResolution='Month',distributionType='Standard'):
    # pdSeries: Pandas series with datetimeindex
    # numYears: Baseline will be based on +/- numYears rolling mean (i.e. for numYears=5, a ten-year centered baseline is calculated)
    # timeResolution: 
    #   Time-resolution at which data should be aggregated and that the baseline should be calculated at. 
    #   Supports "Year", "Month", "Week" and "Day". 
    #   For "Week", estimates for week 52 is used for week 53
    #   For "Day" average of February 28th and March 1st is used for leapday
    # distributionType: Type of distribution to assume for uncertainty. Supports "Standard" and "Poisson". 
    #   "Standard": Calculates standard deviation of data-points used for mean as sqrt(E(x^2) - E(x)^2)
    #   "Poisson": Returns poisson.logsf() of the baseline (i.e. the logarithm of the survival-function given the calculated baseline and the observed data)
    #   Note that baseline is the same for Standard and Poisson
    # Outputs:
    #   curMean: Baseline
    #   curUncertainty: Either standard deviation of log-survival function, depending on the chosen distributionstype
    validTimeResolutions = ["Year", "Month", "Week", "Day"]
    if (timeResolution not in validTimeResolutions):
        raise NameError(f'The timeResolution "{timeResolution}" is not a valid time resolution. Valid options are {validTimeResolutions}')
    
    # Restructure series into pivottable (based on timeResolution)
    curPivot = seriesToPivot(pdSeries,timeResolution)

    # Calculate sum of surrounding years and current year
    curRolling = curPivot.rolling(window=(numYears*2)+1,center=True,min_periods=1)
    curSum = curRolling.sum() # Get sum of all values in roll
    curCount = curRolling.count() # Count how many values were used in sum (to avoid counting NaN's)    
    # Calculate mean of surrounding years by subtracting the current year and dividing by the number of surrounding years 
    # (Replace NaN values with 0. Since the number of Non-NaN values are already counted and used as the divisor, this is fine)
    curBase = (curSum - curPivot.fillna(0))/(curCount-curPivot.notna()*1)

    ### Determine uncertainty
    if distributionType == 'Standard':
        # Calculate the sum of squares of surrounding years and current year
        curSumSqr = curPivot.pow(2).rolling(window=(numYears*2)+1,center=True,min_periods=1).sum()
        curBaseSqr = (curSumSqr - curPivot.pow(2).fillna(0))/(curCount-curPivot.notna()*1)

        # Calculate emperical standard deviation 
        curUncertainty = (curBaseSqr - curBase.pow(2).fillna(0)).pow(0.5)

    elif distributionType == 'Poisson':
        curUncertainty = pd.DataFrame(poisson.logsf(curPivot,curBase),columns=curPivot.columns,index=curPivot.index)

    # For daily time-resolution, everything is also calculated for leap days in non-leap years. Instead, the average of surrounding days is a better estimate
    if timeResolution == 'Day':
        # For leap days, use the average of February 28th and March 1st (Leap-days in non-leap-years will be removed below anyways)
        curBase.loc[:,(2,29)] = (curBase.loc[:,(2,28)] + curBase.loc[:,(3,1)])/2
        curUncertainty.loc[:,(2,29)] = (curUncertainty.loc[:,(2,28)] + curUncertainty.loc[:,(3,1)])/2

    # For weekly time-resolution, use values calculated for week 52 in week 53
    if timeResolution == 'Week':
        curBase[53] = curBase[52] 
        curUncertainty[53] = curUncertainty[53]

    # Reshape pivottables into series
    curBase = reshapePivot(curBase,timeResolution=timeResolution).rename('Baseline')
    curUncertainty  = reshapePivot(curUncertainty,timeResolution=timeResolution)

    # Rename the uncertainty according to distributiontype
    uncertaintyNameDict = {
        'Standard':'StandardDeviation',
        'Poisson':'LogSurvivalFunction',
    }
    curUncertainty  = curUncertainty.rename(uncertaintyNameDict[distributionType])

    return curBase,curUncertainty 

def getPoissonIntervals(intervalValue,curBase):
    # Helper function for getting the probability intervals when assuming a poisson distribution
    # Calculates the top and bottom of the "inner" interval, and returns it as a series with same indices as the baseline
    curBot,curTop = poisson.interval(intervalValue,curBase)
    curBot = pd.Series(curBot,index=curBase.index)
    curTop = pd.Series(curTop,index=curBase.index)
    return curBot,curTop 

def getExcessAndZscore(pdSeries,curBase,curStd):
    # Calculates excess mortality, Z-score and excess mortality in percent

    # Calculate excess as difference between mean and data
    curExc = pdSeries - curBase 
    # And Z-score as excess in terms of standard deviations 
    curZsc = curExc / curStd 

    # Calculate the excess mortality in percent above baseline
    curExcPct = 100 * curExc/curBase 

    return curExc,curZsc,curExcPct






def removeAboveThresholdPoisson(pdSeries,curSF,intervalValue=None,ZscoreThreshold=3):

    if intervalValue == None:
        # If no intervalue is given, use the ZscoreThreshold value. Otherwise ZscoreThreshold isn't used in function.
        intervalValue = norm.cdf(ZscoreThreshold)

    dataToReturn = pdSeries.copy() 
    dataToReturn.loc[curSF < np.log(1-intervalValue)] = np.nan 

    return dataToReturn

def removeAboveThresholdPoissonAndRecalculate(pdSeries,curSF,intervalValue=None,ZscoreThreshold=3,numYears=5,timeResolution='Month'):
    
    curData = removeAboveThresholdPoisson(pdSeries,curSF,intervalValue=intervalValue,ZscoreThreshold=ZscoreThreshold)

    curMean,curSF = rnMean(curData,numYears=numYears,timeResolution=timeResolution,distributionType='Poisson')

    return curData,curMean,curSF

def removeAboveThresholdPoissonAndRecalculateRepeat(pdSeries,curBaseline,curSurvivalFunction,intervalValue=None,ZscoreThreshold=3,numYears=5,timeResolution='Month',verbose=False):
    # Iteratively sets data outside a given interval or ZscoreThreshold to NaN and recalculates baseline and survivalfunction, until all datapoints above threshold is removed.
    # pdSeries should be aggregated correctly before using this function
    # Returns raw data, final baseline and final standard deviation.
    
    if intervalValue == None:
        # If no intervalue is given, use the ZscoreThreshold value. Otherwise ZscoreThreshold isn't used in function.
        intervalValue = norm.cdf(ZscoreThreshold)

    curData = pdSeries.copy()
    curDataRemove = curData.copy()

    numAboveThreshold = 1
    # Count number of iterations (for printing)
    numIter = 0

    while numAboveThreshold > 0:
        # Determine number of entries above threshold
        # _,curZsc,_ = getExcessAndZscore(curDataRemove,curBaseline,curStandardDeviation)
        # numAboveThreshold = (curZsc > ZscoreThreshold).sum()
        numAboveThreshold = (curSurvivalFunction < np.log(1-intervalValue)).sum()

        if verbose:
            # Increment counter
            numIter += 1
            print(f'Iteration {numIter} of removing larger crises. {numAboveThreshold} found.')
            # print(f'Count above threshold: {numAboveThreshold}')


        curDataRemove,curBaseline,curSurvivalFunction = removeAboveThresholdPoissonAndRecalculate(curDataRemove,curSurvivalFunction,intervalValue=None,ZscoreThreshold=ZscoreThreshold,numYears=numYears,timeResolution=timeResolution)

    # # Once everything has been removed, recalculate mean and std with original data
    # curBaseline,curSurvivalFunction = rnMean(curDataRemove,numYears=numYears,timeResolution=timeResolution)


    return curData,curBaseline,curSurvivalFunction 

def removeAboveThresholdPoissonAndRecalculateRepeatFull(pdSeries,intervalValue=None,ZscoreThreshold=3,numYears=5,timeResolution='Month',verbose=False):
    # Calculates baseline and survivalfunction and runs the removeAboveThresholdAndRecalculateRepeat function (see above)
    # pdSeries should be aggregated correctly before using this function
    # Returns raw data, final baseline and final standard deviation.
    
    curBaseline,curSurvivalFunction = rnMean(pdSeries,numYears=numYears,timeResolution=timeResolution,distributionType='Poisson')


    curData,curBaseline,curSurvivalFunction = removeAboveThresholdPoissonAndRecalculateRepeat(pdSeries,curBaseline,curSurvivalFunction,intervalValue=intervalValue,ZscoreThreshold=ZscoreThreshold,numYears=numYears,timeResolution=timeResolution,verbose=verbose)

    return curData,curBaseline,curSurvivalFunction 

def runFullAnalysisDailySeriesPoisson(pdSeries,numYears = 12,intervalValue=None,ZscoreThreshold=3,verbose=False):
    # Assumes pdSeries has datetime64 as index 
    # Note that if data has to be averaged by week (e.g. because sundays are more common as burial days than any other weekday), this should be done *before* running this function.

    # Make a copy, to avoid overwriting things
    pdSeries = pdSeries.copy()

    # Run analysis of all data
    _,curBaseline,curSurvivalFunction = removeAboveThresholdPoissonAndRecalculateRepeatFull(pdSeries,intervalValue=intervalValue,ZscoreThreshold=ZscoreThreshold,numYears=numYears,timeResolution='Day',verbose=verbose)

    # Also calculate the residuals with the corrected baseline
    curExcess = pdSeries - curBaseline 
    curExcessPct = 100 * curExcess/curBaseline

    # Return everything
    return curBaseline,curSurvivalFunction,curExcess,curExcessPct









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

def determineMortalityCrisis(curTime,curExcess,curZscore,upperThreshold=3,lowerThreshold=2,maxDaysBelowThreshold=7,minDurationOfCrisis=0,returnExcessCount=False):
    # --- General function for identifying mortality crises ---
    # Method:
    #   Determines all points where 'curZscore' is above 'upperThreshold'.
    #   For each point, the start-date (and end-date) of the crisis is determined by going backward (and forwards) in time until 'maxDaysBelowThreshold' consecutive days below 'lowerThreshold' has been reached.
    #   The first and last dates within each such group is then considered part of the mortality crisis, and the sum of excess deaths in the period is calculated.
    #   Mortality crises with a duration below minDurationOfCrisis is removed before returning list of mortality crises.   
    # --Inputs--
    #   curTime: datetime array (days)
    #   curExcess: Daily excess mortality (count)
    #   curZscore: Daily excess in number of Z-scores
    #   upperThreshold: Z-score threshold that has to be exceed for something to be considered a crisis
    #   lowerThreshold: A crisis is considered over when it has been lower than this threshold (in Z-scores) for "maxDaysBelowThreshold" number of days
    #   maxDaysBelowThreshold: See above (Max days below threshold to allow groups to be connected)
    #   minDurationOfCrisis: Crises with a shorter duration than this number of days are omitted
    #   returnExcessCount: Boolean flag for returning total number of excess deaths in crisisperiod.
    # -- Returns --
    #   dateGroups: List of tuples with start-date and end-date
    #   allExcess: List with sum of excess deaths in periods given in dateGroups. Only returned if returnExcessCount is set to True

    # Determine all dates above threshold
    curCrisisIndicies = np.where(curZscore > upperThreshold)[0]

    # Initialize array for groupings
    allGroupings = []
    
    # Group and remove crises until there are no left 
    while len(curCrisisIndicies) > 0:

        # Get the index of the first crisis
        curIndex = curCrisisIndicies[0]
        
        ### Going forward
        curBuffer = 0
        thisIndex = curIndex 
        # Default: The next value is below threshold
        IndexEnd_FirstBelow = thisIndex + 1
        # Until the buffer exceeds max
        while curBuffer < maxDaysBelowThreshold:
            # Increase the index to check by one
            thisIndex += 1
            
            # Check if the index exceeds the max possible index (since it would loop around otherwise)
            if thisIndex < len(curZscore):
                # Check if the value is below the threshold
                if curZscore[thisIndex] < lowerThreshold:
                    # If below, increase buffer by one
                    curBuffer += 1
                else:
                    # If above, reset buffer
                    curBuffer = 0 
                    # and set the last possible to the next
                    IndexEnd_FirstBelow = thisIndex +1
            else:
                # If the end of the last has been reached, set the buffer to max to end while-loop
                curBuffer = maxDaysBelowThreshold
                # And reduce the last index by one
                IndexEnd_FirstBelow -= 1



        ### Go backward
        curBuffer = 0
        thisIndex = curIndex 
        # Default: This index is the first value above threshold
        IndexStart_FirstAbove = thisIndex 

        # If the current index is the start of the array, skip to avoid looping around to end of the array
        if thisIndex == 0:
            curBuffer = maxDaysBelowThreshold

        # Until the buffer exceeds max
        while curBuffer < maxDaysBelowThreshold:
            # Go one backward in array
            thisIndex -= 1
            # Check if the value is below the threshold
            if curZscore[thisIndex] < lowerThreshold:
                # If below, increase buffer
                curBuffer += 1
            else:
                # If above, reset buffer
                curBuffer = 0 
                # And set the index of the start of the period to the current index
                IndexStart_FirstAbove = thisIndex 
            # If the start of the list is reached, set the buffer to max to continue
            if thisIndex == 0:
                curBuffer = maxDaysBelowThreshold

        ### Determine full range
        # Define the range from the start to the end
        rangeToAdd = np.arange(IndexStart_FirstAbove,IndexEnd_FirstBelow+1) # Range should be one longer, so "IndexEnd_FirstBelow" is also included.

        # Add group to groupings
        allGroupings.append(rangeToAdd) 

        # Remove every value in group from crisis-indices
        whereToDrop = np.where(np.in1d(curCrisisIndicies,rangeToAdd))[0]
        curCrisisIndicies = np.delete(curCrisisIndicies,whereToDrop)

    ## Finished determining all possible crisis-groups
        
    # Remove groups with a duration below the "minDurationOfCrisis" threshold
    allGroupings = [x for x in allGroupings if len(x)> minDurationOfCrisis ]
    # And sort in order of length
    allGroupings.sort(key=len,reverse=True)

    # Sort groupings and count excess
    allExcess = np.zeros(len(allGroupings)) 
    # Go through each group
    for i,gr in enumerate(allGroupings):

        # Determine excess as sum of residuals
        ExcessSum = np.sum(curExcess[gr])
        # Determine the number of days between first and last value (add one because of indexing)
        numDaysAbove = curTime[gr[-1]] - curTime[gr[0]] + np.timedelta64(1,'D')
        # Give number of days as a int instead of a datetime64 object
        numDaysAbove = int(numDaysAbove / np.timedelta64(1,'D'))

        # Save values to array
        allExcess[i] = ExcessSum

    # Sort according to number of excess deaths
    newOrder = np.argsort(allExcess)[::-1]

    # Reorder lists to return
    allExcess = allExcess[newOrder]

    # Reorder groups
    sortGroupings = []
    for i in newOrder:
        sortGroupings.append(allGroupings[i])

    # Define the arrays to return: Just the start and the end dates
    dateGroups = []
    for gr in sortGroupings:
        dateStart = curTime[gr[0]]
        dateEnd = curTime[gr[-1]]

        dateGroups.append([dateStart,dateEnd])

    # Return the groups of dates, and, if returnExcessCount is set to True, also the number of excess deaths within range
    if returnExcessCount:
        return dateGroups,allExcess
    else:
        return dateGroups
