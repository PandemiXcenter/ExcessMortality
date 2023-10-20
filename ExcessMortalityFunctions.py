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

# Function for calculating running mean from surrounding data
# Assumes input is pandas series, and uses pandas 'rolling' to determine mean
# Index should be a datetimeindex, with correct dates
def rnMean(pdSeries,numYears=5,timeResolution='Month'):

    if timeResolution == 'Year':
        # Start by grouping data by year, in case it's daily or monthly resolution
        # (the min_count flag makes sure that if all are NaN, NaN is actually used instead of 0)
        serYear = pdSeries.groupby(pdSeries.index.year.rename('Year')).sum(min_count=1)

        # Calculate sum of surrounding years and current year
        curRolling = serYear.rolling(window=(numYears*2)+1,center=True,min_periods=1)
        curSum = curRolling.sum()
        curCount = curRolling.count()
        # Calculate mean of surrounding years by subtracting the current year and dividing by the number of surrounding years
        # curMean = (curSum - serYear)/(curCount-1)
        curMean = (curSum - serYear.fillna(0))/(curCount-serYear.notna()*1)
        
        # Calculate the sum of squares of surrounding years and current year
        curSumSqr = serYear.pow(2).rolling(window=(numYears*2)+1,center=True,min_periods=1).sum()
        # curMeanSqr = (curSumSqr - serYear.pow(2))/(numYears*2)
        curMeanSqr = (curSumSqr - serYear.pow(2).fillna(0))/(curCount-serYear.notna()*1)

        # Calculate emperical standard deviation 
        curStd = (curMeanSqr - curMean.pow(2)).pow(0.5)
        
        # # Reshape pivottables into series
        # curMean = curMean.reset_index()
        # curMean['Date'] = pd.to_datetime(dict(year=curMean.Year,month=np.ones(len(curMean.Year)),day=np.ones(len(curMean.Year))))
        # curMean = curMean.sort_values('Date').set_index('Date').drop(columns=['Year']).rename(columns={'value':'Baseline'}).iloc[:,0]
        # curStd = curStd.reset_index()
        # curStd['Date'] = pd.to_datetime(dict(year=curStd.Year,month=np.ones(len(curStd.Year)),day=np.ones(len(curStd.Year))))
        # curStd = curStd.sort_values('Date').set_index('Date').drop(columns=['Year']).rename(columns={'value':'StandardDeviation'}).iloc[:,0]

        # curMean = reshapePivot(curMean,timeResolution=timeResolution).rename('Baseline')
        # curStd  = reshapePivot(curStd,timeResolution=timeResolution).rename('StandardDeviation')


    elif timeResolution == 'Month':
        # Start by grouping data by month, in case it's on daily resolution
        serMonth = pdSeries.groupby([pdSeries.index.year.rename('Year'),pdSeries.index.month.rename('Month')]).sum(min_count=1)

        # Organize as pivot table
        curPivot = serMonth.to_frame().pivot_table(serMonth.name,index='Year',columns='Month')

        # Calculate sum of surrounding years and current year
        curRolling = curPivot.rolling(window=(numYears*2)+1,center=True,min_periods=1)
        curSum = curRolling.sum()
        curCount = curRolling.count()
        # curSum = curPivot.rolling(window=(numYears*2)+1,center=True).sum()

        # Calculate mean of surrounding years by subtracting the current year and dividing by the number of surrounding years
        # curMean = (curSum - curPivot)/(numYears*2)
        curMean = (curSum - curPivot.fillna(0))/(curCount-curPivot.notna()*1)
        
        # Calculate the sum of squares of surrounding years and current year
        curSumSqr = curPivot.pow(2).rolling(window=(numYears*2)+1,center=True,min_periods=1).sum()
        # curMeanSqr = (curSumSqr - curPivot.pow(2))/(numYears*2)
        curMeanSqr = (curSumSqr - curPivot.pow(2).fillna(0))/(curCount-curPivot.notna()*1)

        # Calculate emperical standard deviation 
        curStd = (curMeanSqr - curMean.pow(2)).pow(0.5)

        # # Reshape pivottables into series
        # curMean = curMean.reset_index().melt(id_vars='Year')
        # curMean['Date'] = pd.to_datetime(dict(year=curMean.Year,month=curMean.Month,day=np.ones(len(curMean.Year))))
        # curMean = curMean.sort_values('Date').set_index('Date').drop(columns=['Year','Month']).rename(columns={'value':'Baseline'}).iloc[:,0]
        # curStd = curStd.reset_index().melt(id_vars='Year')
        # curStd['Date'] = pd.to_datetime(dict(year=curStd.Year,month=curStd.Month,day=np.ones(len(curStd.Year))))
        # curStd = curStd.sort_values('Date').set_index('Date').drop(columns=['Year','Month']).rename(columns={'value':'StandardDeviation'}).iloc[:,0]

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
        curPivot = curFrame.pivot_table(values=pdSeries.name,columns=['Month','Day'],index='Year')


        # Calculate sum of surrounding years and current year
        curRolling = curPivot.rolling(window=(numYears*2)+1,center=True,min_periods=1)
        curSum = curRolling.sum() # Get sum of all values in roll
        curCount = curRolling.count() # Count how many values were used in sum (to avoid counting NaN's)
        # Calculate mean of surrounding years by subtracting the current year and dividing by the number of surrounding years 
        # (Replace NaN values with 0. Since the number of Non-NaN values are already counted and used as the divisor, this is fine)
        # curMean = (curSum - curPivot.fillna(0))/(curCount-1)
        curMean = (curSum - curPivot.fillna(0))/(curCount-curPivot.notna()*1)

        # # Calculate sum of surrounding years and current year
        # curSum = curPivot.rolling(window=(numYears*2)+1,center=True).sum()
        # # Calculate mean of surrounding years by subtracting the current year and dividing by the number of surrounding years
        # curMean = (curSum - curPivot)/(numYears*2)
        
        # Calculate the sum of squares of surrounding years and current year
        curRollingSqr = curPivot.pow(2).rolling(window=(numYears*2)+1,center=True,min_periods=1)
        curSumSqr = curRollingSqr.sum()
        # curMeanSqr = (curSumSqr - curPivot.pow(2).fillna(0))/(curCount-1)
        curMeanSqr = (curSumSqr - curPivot.pow(2).fillna(0))/(curCount-curPivot.notna()*1)


        # curSumSqr = curPivot.pow(2).rolling(window=(numYears*2)+1,center=True).sum()
        # curMeanSqr = (curSumSqr - curPivot.pow(2))/(numYears*2)

        # Calculate emperical standard deviation 
        curStd = (curMeanSqr - curMean.pow(2).fillna(0)).pow(0.5)

        # For leap days, use the average of February 28th and March 1st (Leap-days in non-leap-years will be removed below anyways)
        curMean.loc[:,(2,29)] = (curMean.loc[:,(2,28)] + curMean.loc[:,(3,1)])/2
        curStd.loc[:,(2,29)] = (curStd.loc[:,(2,28)] + curStd.loc[:,(3,1)])/2

        # # Reshape pivottables into series
        # curMean = curMean.reset_index().melt(id_vars='Year') # Melt pivottable
        # curMean['Date'] = pd.to_datetime(dict(year=curMean.Year,month=curMean.Month,day=curMean.Day),errors='coerce') # Make a date-columns (coerce "false" leap-days to NaT, i.e. in non-leap years)
        # curMean = curMean.sort_values('Date').set_index('Date').drop(columns=['Year','Month','Day']).rename(columns={'value':'Baseline'}).iloc[:,0] # Sort by date and drop extra columns
        # curMean = curMean.loc[curMean.index.notna()] # Remove invalid dates (leap-days in not leap-years)

        # curStd = curStd.reset_index().melt(id_vars='Year') # Melt pivottable
        # curStd['Date'] = pd.to_datetime(dict(year=curStd.Year,month=curStd.Month,day=curStd.Day),errors='coerce') # Make a date-columns (coerce "false" leap-days to NaT, i.e. in non-leap years)
        # curStd = curStd.sort_values('Date').set_index('Date').drop(columns=['Year','Month','Day']).rename(columns={'value':'StandardDeviation'}).iloc[:,0] # Sort by date and drop extra columns
        # curStd = curStd.loc[curStd.index.notna()] # Remove invalid dates (leap-days in not leap-years)


    # Reshape pivottables into series
    curMean = reshapePivot(curMean,timeResolution=timeResolution).rename('Baseline')
    curStd  = reshapePivot(curStd,timeResolution=timeResolution).rename('StandardDeviation')

    return curMean,curStd 

def getExcessAndZscore(pdSeries,curMean,curStd):

    # Calculate excess as difference between mean and data
    curExc = pdSeries - curMean 
    # And Z-score as excess in terms of standard deviations 
    curZsc = curExc / curStd 

    return curExc,curZsc

def removeAboveThreshold(pdSeries,curMean,curStd,ZscoreThreshold=3):

    curExc,curZsc = getExcessAndZscore(pdSeries,curMean,curStd)

    dataToReturn = pdSeries.copy()
    dataToReturn.loc[curZsc[curZsc > ZscoreThreshold].index] = np.nan 

    return dataToReturn

def removeAboveThresholdAndRecalculate(pdSeries,curMean,curStd,ZscoreThreshold=3,numYears=5,timeResolution='Month'):

    curData = removeAboveThreshold(pdSeries.copy(),curMean,curStd,ZscoreThreshold=ZscoreThreshold)

    curMean,curStd = rnMean(curData,numYears=numYears,timeResolution=timeResolution)

    return curData,curMean,curStd 

def removeAboveThresholdAndRecalculateRepeat(pdSeries,curMean,curStd,ZscoreThreshold=3,numYears=5,timeResolution='Month'):

    curData = pdSeries.copy()
    curDataRemove = curData.copy()

    numAboveThreshold = 1

    while numAboveThreshold > 0:
        # Determine number of entries above threshold
        curExc,curZsc = getExcessAndZscore(curDataRemove,curMean,curStd)
        numAboveThreshold = (curZsc > ZscoreThreshold).sum()
        print(numAboveThreshold)


        curDataRemove,curMean,curStd = removeAboveThresholdAndRecalculate(curDataRemove,curMean,curStd,ZscoreThreshold=ZscoreThreshold,numYears=numYears,timeResolution=timeResolution)

    # # Once everything has been removed, recalculate mean and std with original data
    # curMean,curStd = rnMean(curDataRemove,numYears=numYears,timeResolution=timeResolution)


    return curData,curMean,curStd 

