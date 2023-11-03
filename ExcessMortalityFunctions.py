# --- Import dependencies ---
import numpy as np 
import pandas as pd 
import math 
from scipy.stats import poisson


# Simple functions for running mean, using convolution. Input should be a numpy array
def rnMeanSimple(data,meanWidth=7):
    return np.convolve(data, np.ones(meanWidth)/meanWidth, mode='valid')
def rnTimeSimple(t,meanWidth=7):
    return t[math.floor(meanWidth/2):-math.ceil(meanWidth/2)+1]


def groupByYear(pdSeries):
    # Groups data by year
    pdSeries = pdSeries.groupby([pdSeries.index.year.rename('Year')]).sum().reset_index()

    pdSeries['Date'] = pd.to_datetime(dict(year=pdSeries.Year,month=np.ones(len(pdSeries.Year)),day=np.ones(len(pdSeries.Year))))

    pdSeries = pdSeries.drop(columns='Year').set_index('Date').iloc[:,0]
    
    return pdSeries

def groupByMonth(pdSeries):
    # Groups data by month
    pdSeries = pdSeries.groupby([pdSeries.index.year.rename('Year'),pdSeries.index.month.rename('Month')]).sum().reset_index()

    pdSeries['Date'] = pd.to_datetime(dict(year=pdSeries.Year,month=pdSeries.Month,day=np.ones(len(pdSeries.Year))))

    pdSeries = pdSeries.drop(columns=['Year','Month']).set_index('Date').iloc[:,0]
    
    return pdSeries
    
def groupByWeek(pdSeries):
    # Groups data by week
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
        curPivot = curFrame.pivot_table(values=curFrame.columns[0],index='Year',columns=['Month','Day'])

    return curPivot

# Function for calculating running mean from surrounding data
# Assumes input is pandas series, and uses pandas 'rolling' to determine mean
# Index should be a datetimeindex, with correct dates
def rnMean(pdSeries,numYears=5,timeResolution='Month',DistributionType='Standard'):

    # Restructure series into pivottable (based on timeResolution)
    curPivot = seriesToPivot(pdSeries,timeResolution)

    # Calculate sum of surrounding years and current year
    curRolling = curPivot.rolling(window=(numYears*2)+1,center=True,min_periods=1)
    curSum = curRolling.sum() # Get sum of all values in roll
    curCount = curRolling.count() # Count how many values were used in sum (to avoid counting NaN's)    
    # Calculate mean of surrounding years by subtracting the current year and dividing by the number of surrounding years 
    # (Replace NaN values with 0. Since the number of Non-NaN values are already counted and used as the divisor, this is fine)
    curMean = (curSum - curPivot.fillna(0))/(curCount-curPivot.notna()*1)

    ### Determine uncertainty
    if DistributionType == 'Standard':
        # Calculate the sum of squares of surrounding years and current year
        curSumSqr = curPivot.pow(2).rolling(window=(numYears*2)+1,center=True,min_periods=1).sum()
        curMeanSqr = (curSumSqr - curPivot.pow(2).fillna(0))/(curCount-curPivot.notna()*1)

        # Calculate emperical standard deviation 
        curStd = (curMeanSqr - curMean.pow(2).fillna(0)).pow(0.5)

    elif DistributionType == 'Poisson':
        curSF = pd.DataFrame(poisson.logsf(curPivot,curMean),columns=curPivot.columns,index=curPivot.index)

    # For daily time-resolution, everything is also calculated for leap days in non-leap years. Instead, the average of surrounding days is a better estimate
    if timeResolution == 'Day':
        # For leap days, use the average of February 28th and March 1st (Leap-days in non-leap-years will be removed below anyways)
        curMean.loc[:,(2,29)] = (curMean.loc[:,(2,28)] + curMean.loc[:,(3,1)])/2
        
        if DistributionType == 'Standard':
            curStd.loc[:,(2,29)] = (curStd.loc[:,(2,28)] + curStd.loc[:,(3,1)])/2
        elif DistributionType == 'Poisson':
            curSF.loc[:,(2,29)] = (curSF.loc[:,(2,28)] + curSF.loc[:,(3,1)])/2

    # For weekly time-resolution, use values calculated for week 52 in week 53
    if timeResolution == 'Week':
        curMean[53] = curMean[52] 

        if DistributionType == 'Standard':
            curStd[53] = curStd[53]
        elif DistributionType == 'Poisson':
            curSF[53] = curSF[53]

    # Reshape pivottables into series
    curMean = reshapePivot(curMean,timeResolution=timeResolution).rename('Baseline')
    
    if DistributionType == 'Standard':
        curStd  = reshapePivot(curStd,timeResolution=timeResolution).rename('StandardDeviation')
    elif DistributionType == 'Poisson':
        curSF  = reshapePivot(curSF,timeResolution=timeResolution).rename('LogSurvivalFunction')

    if DistributionType == 'Standard':
        return curMean,curStd 
    elif DistributionType == 'Poisson':
        return curMean,curSF

def getPoissonIntervals(intervalValue,curBase):
    # Helper function for getting the probability intervals when assuming a poisson distribution
    # Calculates the top and bottom of the "inner" interval, and returns it as a series with same indices as the baseline
    curBot,curTop = poisson.interval(intervalValue,curBase)
    curBot = pd.Series(curBot,index=curBase.index)
    curTop = pd.Series(curTop,index=curBase.index)
    return curBot,curTop 

def getExcessAndZscore(pdSeries,curBase,curStd):

    # Calculate excess as difference between mean and data
    curExc = pdSeries - curBase 
    # And Z-score as excess in terms of standard deviations 
    curZsc = curExc / curStd 

    # Calculate the excess mortality in percent above baseline
    curExcPct = 100 * curExc/curBase 

    return curExc,curZsc,curExcPct


def removeAboveThreshold(pdSeries,curMean,curStd,ZscoreThreshold=3):

    curExc,curZsc,curExcPct = getExcessAndZscore(pdSeries,curMean,curStd)

    dataToReturn = pdSeries.copy()
    dataToReturn.loc[curZsc[curZsc > ZscoreThreshold].index] = np.nan 

    return dataToReturn

def removeAboveThresholdAndRecalculate(pdSeries,curMean,curStd,ZscoreThreshold=3,numYears=5,timeResolution='Month'):

    curData = removeAboveThreshold(pdSeries.copy(),curMean,curStd,ZscoreThreshold=ZscoreThreshold)

    curMean,curStd = rnMean(curData,numYears=numYears,timeResolution=timeResolution)

    return curData,curMean,curStd 

def removeAboveThresholdAndRecalculateRepeat(pdSeries,curMean,curStd,ZscoreThreshold=3,numYears=5,timeResolution='Month',verbose=False):

    curData = pdSeries.copy()
    curDataRemove = curData.copy()

    numAboveThreshold = 1
    # Count number of iterations (for printing)
    numIter = 0

    while numAboveThreshold > 0:
        # Increment counter
        numIter += 1
        # Determine number of entries above threshold
        curExc,curZsc,curExcPct = getExcessAndZscore(curDataRemove,curMean,curStd)
        numAboveThreshold = (curZsc > ZscoreThreshold).sum()

        if verbose:
            print(f'Finished iteration {numIter+1} of removing larger crises. {numAboveThreshold} found.')
            # print(f'Count above threshold: {numAboveThreshold}')


        curDataRemove,curMean,curStd = removeAboveThresholdAndRecalculate(curDataRemove,curMean,curStd,ZscoreThreshold=ZscoreThreshold,numYears=numYears,timeResolution=timeResolution)

    # # Once everything has been removed, recalculate mean and std with original data
    # curMean,curStd = rnMean(curDataRemove,numYears=numYears,timeResolution=timeResolution)


    return curData,curMean,curStd 

def removeAboveThresholdAndRecalculateRepeatFull(pdSeries,ZscoreThreshold=3,numYears=5,timeResolution='Month',verbose=False):

    curMean,curStd = rnMean(pdSeries,numYears=numYears,timeResolution=timeResolution,DistributionType='Standard')

    curData = pdSeries.copy()
    curDataRemove = curData.copy()

    numAboveThreshold = 1

    # Count number of iterations (for printing)
    numIter = 0

    while numAboveThreshold > 0:
        # Increment counter
        numIter += 1
        # Determine number of entries above threshold
        curExc,curZsc,curExcPct = getExcessAndZscore(curDataRemove,curMean,curStd)
        numAboveThreshold = (curZsc > ZscoreThreshold).sum()

        if verbose:
            print(f'Finished iteration {numIter+1} of removing larger crises. {numAboveThreshold} found.')
            # print(f'Count above threshold: {numAboveThreshold}')


        curDataRemove,curMean,curStd = removeAboveThresholdAndRecalculate(curDataRemove,curMean,curStd,ZscoreThreshold=ZscoreThreshold,numYears=numYears,timeResolution=timeResolution)

    # # Once everything has been removed, recalculate mean and std with original data
    # curMean,curStd = rnMean(curDataRemove,numYears=numYears,timeResolution=timeResolution)


    return curData,curMean,curStd 


def runFullAnalysisOneDataframeDaily(curdf,columnsToUse=['Total'],numYears = 12,ZscoreThreshold=3,verbose=False):
    ##### NO LONGER USED, Can be removed... 

    # Assumes curdf has datetime64 as index 

    # Make a copy, to avoid overwriting things
    curdf = curdf.copy()
    
    # Save the date to an array to return
    this_curTime = curdf.index

    # Sum columns to use and calculate 7-day rolling (to avoid sunday-trouble)
    this_curVals = curdf[columnsToUse].sum(axis=1).copy() # Use the sum of the columns input
    this_curVals = this_curVals.rolling(window=7,center=True).mean()

    # Run analysis of all data
    newDataNew,this_corrMean,this_corrStd = removeAboveThresholdAndRecalculateRepeatFull(this_curVals,ZscoreThreshold=ZscoreThreshold,numYears=numYears,timeResolution='Day',verbose=verbose)
    # (this_corrVals,this_corrMean,this_corrStd,this_corrResi,this_corrResiStd,this_corrResiPct,this_allCrisisIndices) = removeAllIterativelyFull(this_curVals,numYears = numYears,threshold=thresholdExcess,verbose=verbose)

    # Also calculate the residuals with the corrected baseline
    this_postResi = this_curVals - this_corrMean 
    this_postResiStd = this_postResi/this_corrStd  
    this_postResiPct = 100 * this_postResi/this_corrMean

    # Return everything
    return this_curTime,this_curVals,this_corrMean,this_corrStd,this_postResi,this_postResiStd,this_postResiPct



def runFullAnalysisDailySeries(pdSeries,numYears = 12,ZscoreThreshold=3,verbose=False):
    # Assumes curdf has datetime64 as index 
    # Note that if data has to be averaged by week (e.g. because sundays are more common as burial days than any other weekday), this should be done *before* running this function.

    # Make a copy, to avoid overwriting things
    pdSeries = pdSeries.copy()
    
    # Save the date to an array to return
    this_curTime = pdSeries.index

    # Run analysis of all data
    newDataNew,this_corrMean,this_corrStd = removeAboveThresholdAndRecalculateRepeatFull(pdSeries,ZscoreThreshold=ZscoreThreshold,numYears=numYears,timeResolution='Day',verbose=verbose)
    # (this_corrVals,this_corrMean,this_corrStd,this_corrResi,this_corrResiStd,this_corrResiPct,this_allCrisisIndices) = removeAllIterativelyFull(this_curVals,numYears = numYears,threshold=thresholdExcess,verbose=verbose)

    # Also calculate the residuals with the corrected baseline
    this_postResi = pdSeries - this_corrMean 
    this_postResiStd = this_postResi/this_corrStd  
    this_postResiPct = 100 * this_postResi/this_corrMean

    # Return everything
    return this_curTime,pdSeries,this_corrMean,this_corrStd,this_postResi,this_postResiStd,this_postResiPct
