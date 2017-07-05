setwd("D:\\Users\\argho\\kaggle\\Walmart_Forecasting")

library(forecast)
library(timeDate)
library(randomForest)
#library(neuralnet)

# read
dataset <- read.csv("train.csv",header = T)

# filter by store and department
dataset_store1 = subset(dataset, Store == 1)

# checkign number of rows
nrow(dataset_store1)


# plotting timeseries
dataset_store1_ts = ts(dataset_store1$Weekly_Sales, frequency=52, start = c(2010,5))
plot(dataset_store1_ts)

# decomposition of the dataset
plot(decompose(dataset_store1_ts))


# breaking into training and validation
train_store1 = head(dataset_store1 , 120)
test_store1  = tail(dataset_store1 , nrow(dataset_store1) - nrow(train_store1))

train_store1_ts = ts(train_store1$Weekly_Sales, frequency=52, start = c(2010,5))

# Model 1 - Naive Model
train_store1_naive = naive(train_store1_ts, h = nrow(test_store1))
train_store1_naive_forecast = forecast(train_store1_naive)
accuracy(train_store1_naive_forecast, x = test_store1$Weekly_Sales)

plot(train_store1_naive_forecast)

# Model 5 : Average Method
train_store1_avg_forecast = meanf(train_store1_ts, h = 23)
accuracy(train_store1_avg_forecast, x = test_store1$Weekly_Sales)

plot(train_store1_avg_forecast)

# Model 5 : Random walk ( with and without Drift)
# with drift
train_store1_rw_forecast = rwf(train_store1_ts, h = 23, drift = T)
accuracy(train_store1_rw_forecast, x = test_store1$Weekly_Sales)

plot(train_store1_rw_forecast)

# without drift
train_store1_rwd_forecast = rwf(train_store1_ts, h = 23, drift = F)
accuracy(train_store1_rwd_forecast, x = test_store1$Weekly_Sales)

plot(train_store1_rwd_forecast)

# Model 2 - Seasonal Naive Model
train_store1_snaive = snaive(train_store1_ts)
train_store1_snaive_forecast = forecast(train_store1_snaive, h = nrow(test_store1))
accuracy(train_store1_snaive_forecast, x = test_store1$Weekly_Sales)

plot(train_store1_snaive_forecast)


# Model 3 - Simple Exponential Smoothing
train_store1_ses_model = HoltWinters(train_store1_ts, beta = F, gamma = F)
train_store1_ses_forecast = forecast(train_store1_ses_model, h = nrow(test_store1))
accuracy(train_store1_ses_forecast,  x = test_store1$Weekly_Sales)
plot(train_store1_ses_model)
plot(train_store1_ses_forecast)


acf(train_store1_ses_forecast$residuals[-1], lag.max=23)
pacf(train_store1_ses_forecast$residuals[-1], lag.max=23)

plot.ts(train_store1_ses_forecast$residuals)

plotForecastErrors <- function(forecasterrors)
{
  # make a histogram of the forecast errors:
  mybinsize <- IQR(forecasterrors)/4
  mysd   <- sd(forecasterrors)
  mymin  <- min(forecasterrors) - mysd*5
  mymax  <- max(forecasterrors) + mysd*3
  # generate normally distributed data with mean 0 and standard deviation mysd
  mynorm <- rnorm(10000, mean=0, sd=mysd)
  mymin2 <- min(mynorm)
  mymax2 <- max(mynorm)
  if (mymin2 < mymin) { mymin <- mymin2 }
  if (mymax2 > mymax) { mymax <- mymax2 }
  # make a red histogram of the forecast errors, with the normally distributed data overlaid:
  mybins <- seq(mymin, mymax, mybinsize)
  hist(forecasterrors, col="red", freq=FALSE, breaks=mybins)
  # freq=FALSE ensures the area under the histogram = 1
  # generate normally distributed data with mean 0 and standard deviation mysd
  myhist <- hist(mynorm, plot=FALSE, breaks=mybins)
  # plot the normal curve as a blue line on top of the histogram of forecast errors:
  points(myhist$mids, myhist$density, type="l", col="blue", lwd=2)
}

plotForecastErrors(train_store1_ses_forecast$residuals[-1])



# Model 4 - Holt's Exponential Smoothing

train_store1_ses_model = HoltWinters(train_store1_ts, beta = F, gamma = T)
train_store1_ses_forecast = forecast(train_store1_ses_model, h = nrow(test_store1))
accuracy(train_store1_ses_forecast, x = test_store1$Weekly_Sales)
plot(train_store1_ses_model)
plot(train_store1_ses_forecast)
plot.ts(train_store1_ses_forecast$residuals)


# Model 6 : ARIMA
train_store1_autoArima = auto.arima(train_store1_ts)
train_1_1_ts_autoArima_fc = forecast.Arima(train_store1_autoArima, h = nrow(test_store1))
accuracy(train_1_1_ts_autoArima_fc, x = test_store1$Weekly_Sales)

plot(train_1_1_ts_autoArima_fc)
plot(train_1_1_ts_autoArima_fc$residuals)

# Model 7 : Random Forest
# Random Forest is a general Machine Learning Algorithm and is not specific to TimeSeries
dataset_store1_rf = dataset_store1

# creating new features from existing features of data set
dataset_store1_rf$year = as.numeric(substr(dataset_store1_rf$Date,7,10))
dataset_store1_rf$month = as.numeric(substr(dataset_store1_rf$Date,4,5))
dataset_store1_rf$day = as.numeric(substr(dataset_store1_rf$Date,1,2))
dataset_store1_rf$days = (dataset_store1_rf$month-1)*30 + dataset_store1_rf$day
dataset_store1_rf$IsHoliday[dataset_store1_rf$IsHoliday=="TRUE"]=1
dataset_store1_rf$IsHoliday[dataset_store1_rf$IsHoliday=="FALSE"]=0
dataset_store1_rf$dayHoliday = dataset_store1_rf$IsHoliday*dataset_store1_rf$days
dataset_store1_rf$logsales = log(dataset_store1_rf$Weekly_Sales)

train_rf = head(dataset_store1_rf, 120)
test_rf = tail(dataset_store1_rf, nrow(dataset_store1_rf) - nrow(train_rf))


# build the model
train_store1_rf =  randomForest(logsales ~ year + month + day + days + dayHoliday , 
                         ntree=4800, replace=TRUE, mtry=3, data=train_rf)


# validation of the model
train_store1_rf_prdt = exp(predict(train_store1_rf,test_rf))
accuracy(ts(train_store1_rf_prdt), test_rf$Weekly_Sales)
