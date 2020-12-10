%% ************************************************************* 
%  FM321 - Risk Management and Modelling
%  Michaelmas 2019 - Course Project - Topic 1 - univariate model analysis
%  Date: 20 January 2019
% **************************************************************

%% *** Part 0: Preliminaries ***
clc;                   
clear all;             
clf;                    
close all;

%% *** Part 1: Loading Data from Excel File ***

Data_Stocks            = xlsread('Project Data.xlsx', ...
                          'FM320-FM321 - Project Data', 'E9:Z7546');

Tickers                = {'MSFT', 'XOM ', 'FB  ', 'CVX ', 'GOOG', 'AAPL', 'PFE ', 'JNJ ', 'WFC ', 'JPM ', 'WMT ', 'BAC ', 'VZ  ','T   ', 'HD  ', 'AMZN', 'GOOGL', 'MA  ', 'UNH ', 'V   ', 'SPX '};

% Useful information
LogReturns             = Data_Stocks(:, 1:(end-1));
Dates                  = Data_Stocks(:, end);
NDates                 = size(Dates, 1);
NSecurities            = size(LogReturns, 2);
DaysPerYear            = 252;
NObsBurnIn             = 252;
StartYear              = 1990;
EndYear                = 2019;
DatesForReturns        = Dates;
NDatesForReturns       = NDates;

% Computing squared returns for all stocks
LogReturnsSq           = LogReturns .* LogReturns;

%% *** Part 2: Computing Returns and Statistics, Producing Display ***
% Choose CVX (Chevron Corporation) to work with
% Computing statistics
RetAux             = LogReturns(:, 4);
RetAux             = RetAux(~isnan(RetAux));
AvgRet             = mean(RetAux);
StdDevRet          = std(RetAux);
MaxRet             = max(RetAux);
MinRet             = min(RetAux);
SkewRet            = skewness(RetAux);
KurtRet            = kurtosis(RetAux);

% Converting averages and standard deviations to annual measures
AvgRetAnn              = DaysPerYear * AvgRet;
StdDevRetAnn           = sqrt(DaysPerYear) * StdDevRet;

% Finding the dates for the maximum and minimum return for CVX
RetAux              = LogReturns(:,4);
RowsAux             = find(~isnan(RetAux));
RetAux              = RetAux(RowsAux);
DatesAux            = DatesForReturns(RowsAux);
RowMax              = find(RetAux == MaxRet(1, 1));
RowMin              = find(RetAux == MinRet(1, 1));
DatesMax            = DatesAux(RowMax);
DatesMin            = DatesAux(RowMin);

% Testing for normality using Jarque-Bera test
RetAux              = LogReturns(:, 4);
RetAux              = RetAux(~isnan(RetAux));
[H, P, JBStat]      = jbtest(RetAux);
JB_Stat             = JBStat;
JBPVal              = P;

% Producing a display which contains all the statistics for CVX
RowHeaders             = {'Average', 'Std. Dev.', 'Max', 'Min', ...
                           'Skewness', 'Kurtosis', 'Average (Ann.)', 'Std. Dev. (Ann.)', 'Date Max', 'Date Min', 'JB Stat.', 'JB P-Val.'};
ColHeaders             = [{' ', 'CVX'}];
Statistics             = [AvgRet; StdDevRet; MaxRet; MinRet; SkewRet; KurtRet; AvgRetAnn; StdDevRetAnn; DatesMax; DatesMin; JB_Stat; JBPVal];
DisplayAux             = [RowHeaders' num2cell(Statistics)];
DisplayStatistics      = [ColHeaders; DisplayAux];

%% *** Part 3: Histograms ***

% Creating histogram for log returns of CVX
FigNo                 = 1;

figure(FigNo);
histogram(LogReturns(:, 4));
TitleStr              = ['Histogram of daily returns for ' Tickers{1, 4}];
title(TitleStr);
set(gca, 'XTickMode', 'manual');
set(gca, 'XTickLabel', num2str(100 .* get(gca, 'XTick')', '%.0f%%'))    
FigNo                 = FigNo + 1;

%% *** Part 4: Exploring Autocorrelations ***
% Produce chart with time series of returns for CVX

% Eliminating rows with NaNs and computing start and end date for chart
ChartData         = LogReturns(:, 4);
ChartRows         = find(~isnan(ChartData));
ChartDates        = DatesForReturns(ChartRows);
ChartData         = ChartData(ChartRows);
NDates            = size(ChartDates, 1);
Aux1              = ChartDates(1, 1);
StartDate         = (Aux1 - mod(Aux1, 10000))/10000;
Aux2              = ChartDates(end, 1);
EndDate           = (Aux2 - mod(Aux2, 10000))/10000;
    
% Producing chart as before
figure(FigNo);
TimeLabels        = linspace(StartDate, EndDate, NDates);
    
Axis1             = [StartDate EndDate -0.75 0.75];  
title(['Daily returns for ' Tickers{1, 4}]);
ylabel('Daily Log Returns');
xlabel('Dates');
axis(Axis1);
grid on;
hold on;
plot(TimeLabels, ChartData, 'LineStyle', '-' , ...
                      'LineWidth', 1, 'Color', 'blue');
set(gcf, 'color', 'white');

FigNo                 = FigNo + 1;

% Produce autocorrelation function for returns series
  
% Note that the autocorr function does not handle missing data, so need
% to remove those from the vector before using it
figure(FigNo);
AuxData           = LogReturns(:, 4);
AuxData           = AuxData(~isnan(AuxData));
autocorr(AuxData);
TitleStr          = ['Sample autocorrelation of returns for ' Tickers{1, 4}];
title(TitleStr);
ylabel('Correlation');
xlabel('Order');

FigNo                 = FigNo + 1;

% Produce autocorrelation function for squared returns series
  
% Note that the autocorr function does not handle missing data, so need
% to remove those from the vector before using it
figure(FigNo);
AuxData           = LogReturnsSq(:, 4);
AuxData           = AuxData(~isnan(AuxData));
autocorr(AuxData);
TitleStr          = ['Sample autocorrelation of squared returns for ' Tickers{1, 4}];
title(TitleStr);
ylabel('Correlation');
xlabel('Order');

FigNo                 = FigNo + 1;

%% *** Part 5: Moving Variance ***

% Produce moving volatility for the retun series and chart them

% Creating variables to store variance data
WindowSize            = 50;     % For each day, compute variance on the basis of this 
                                %    many data points, ending the previous day
    
% Eliminating rows with NaNs and computing start and end date for chart
ChartData         = LogReturns(:, 4);
ChartRows         = find(~isnan(ChartData));
ChartDates        = DatesForReturns(ChartRows);
ChartData         = ChartData(ChartRows);
NDates            = size(ChartDates, 1);

% Loop to compute the moving variance series for this security
MovVar            = NaN(NDates, 1);

for j=(WindowSize+1):NDates
        RowsToUse     = (j - WindowSize):(j-1);         % In date i, use period ending in (i-1)
        MovVar(j, 1)  = var(ChartData(RowsToUse, 1));
end

MovVol            = sqrt(MovVar);                   % Moving volatility

% The next section eliminates rows without data from the various matrices
RowsWithData      = find(~isnan(MovVol));           % These are the rows to keep
ReturnsForChart   = ChartData(RowsWithData, 1);
MovVolForChart    = MovVol(RowsWithData, 1);
ChartDates        = ChartDates(RowsWithData, 1);

Aux1              = ChartDates(1, 1);
StartDate         = (Aux1 - mod(Aux1, 10000))/10000;
Aux2              = ChartDates(end, 1);
EndDate           = (Aux2 - mod(Aux2, 10000))/10000;
TimeLabels        = linspace(StartDate, EndDate, NDates - WindowSize);

% Now, we can chart the returns over time, along with +/- 2 Std. Dev. series
figure(FigNo);
Axis1             = [StartDate EndDate -0.75 0.5];
title(['Daily returns and +/- 2 Std. Dev. from Moving Variance for ' Tickers{1, 4}]);
ylabel('Daily returns');
xlabel('Dates');
axis(Axis1);
grid on;
hold on;
plot(TimeLabels, ReturnsForChart, 'LineStyle', '-' , ...
                      'LineWidth', 1, 'Color', 'blue');
plot(TimeLabels, 2*MovVolForChart, 'LineStyle', '-' , ...
                      'LineWidth', 1, 'Color', 'red');
plot(TimeLabels, -2*MovVolForChart, 'LineStyle', '-' , ...
                      'LineWidth', 1, 'Color', 'red');
legend1 = legend('Returns', '+/- 2 Std. Dev.', ...
                         'location','best');              
set(gcf, 'color', 'white');    

FigNo                 = FigNo + 1;
    
%% *** Part 7: QQ Plots ***

% Produce QQ plots for CVX versus normal and t distributions

% Creating a QQ-plot of stock returns against standard normal distribution
Index                 = 4;     % Working with CVX

ReturnsData           = LogReturns(:, Index);
ReturnsData           = ReturnsData(~isnan(ReturnsData));

figure(FigNo);
qqplot(ReturnsData);
title(['QQ-Plot for ' Tickers{1, Index}]);
ylabel(['Historical Sample for ' Tickers{1, Index}]);
xlabel('Standard normal distribution');
FigNo                 = FigNo + 1;

% Creating a QQ-plot of stock returns against t-3 distribution - must first
% create the distribution to plot against returns
tdist                 = makedist('tLocationScale', 'mu', 0, ...
                           'sigma',1, 'nu', 3);
figure(FigNo);
qqplot(ReturnsData, tdist);
title(['QQ-Plot for ' Tickers{1, Index}]);
ylabel(['Historical Sample for ' Tickers{1, Index}]);
xlabel('Quantiles of t(3) distribution');
FigNo                 = FigNo + 1;

% Creating a QQ-plot of stock returns against t-4 distribution - must first
% create the distribution to plot against returns
tdist                 = makedist('tLocationScale', 'mu', 0, ...
                           'sigma',1, 'nu', 4);
figure(FigNo);
qqplot(ReturnsData, tdist);
title(['QQ-Plot for ' Tickers{1, Index}]);
ylabel(['Historical Sample for ' Tickers{1, Index}]);
xlabel('Quantiles of t(4) distribution');
FigNo                 = FigNo + 1;

% Creating a QQ-plot of stock returns against t-5 distribution - must first
% create the distribution to plot against returns
tdist                 = makedist('tLocationScale', 'mu', 0, ...
                           'sigma',1, 'nu', 5);
figure(FigNo);
qqplot(ReturnsData, tdist);
title(['QQ-Plot for ' Tickers{1, Index}]);
ylabel(['Historical Sample for ' Tickers{1, Index}]);
xlabel('Quantiles of t(5) distribution');
FigNo                 = FigNo + 1;

%% *** Part 8: Compute an EWMA volatility for CVX return series ***
Lambda                 = 0.97;

% Will initialize the computation using an unweighted variance of the
% beginning of the sample
EWMAVar                = NaN(NDatesForReturns, 1);

% First, determine the effective sample for the security
RowsAux             = find(~isnan(LogReturns(:, 4)));
StartRowForStock    = min(RowsAux);
InitialSample       = (StartRowForStock:(StartRowForStock + NObsBurnIn - 1))';
   
EWMAVar(StartRowForStock + NObsBurnIn, 1) = var(LogReturns(InitialSample, 4));
   
% Next, run loop to compute variance estimates
for j = (StartRowForStock + NObsBurnIn + 1):NDatesForReturns
        EWMAVar(j, 1)  = Lambda * EWMAVar(j-1, 1) + (1-Lambda) * ...
                              LogReturnsSq(j-1, 4);
end

% Finally, convert into volatility estimates
EWMAVol                = sqrt(EWMAVar);

%% *** Part 9: Estimate ARCH models for CVX return series ***

% Create the models first; for ARCH(2), will use the tarch function from
% the MFE Toolbox
ARCH01Model            = garch(0, 1);
ARCH10Model            = garch(0, 10);

% Estimate ARCH for the series, save information for computing estimates
ARCH01Info             = NaN(4, 1);
ARCH02Info             = NaN(12, 1);
ARCH10Info             = NaN(14, 1);


ARCHData            = LogReturns(:, 4);
ARCHData            = ARCHData(~isnan(ARCHData));
   
% First, estimate ARCH(1)
EstimatedModel      = estimate(ARCH01Model, ARCHData);
ARCH01Info(1, 1)    = EstimatedModel.Constant;
ARCH01Info(2, 1)    = EstimatedModel.ARCH{1};
ARCH01Info(3, 1)    = EstimatedModel.UnconditionalVariance;
ARCH01Info(4, 1)    = sqrt(DaysPerYear) * sqrt(EstimatedModel.UnconditionalVariance);    
   
% Next, estimate ARCH(2).  We will use the tarch function from the MFE
% Toolbox, in order to obtain t-statistics as requested.
[Parms, ~, ~, VCV]  = tarch(ARCHData, 2, 0, 0);
Omega               = Parms(1, 1);
Alpha1              = Parms(2, 1);
Alpha2              = Parms(3, 1);
   
ParmStdError        = sqrt(diag(VCV));
ParmTStat           = Parms ./ ParmStdError;
ParmPVal            = 2*(1 - normcdf(ParmTStat));

ARCH02Info(1:3, 1)  = Parms;
ARCH02Info(4, 1)    = Alpha1 + Alpha2;
ARCH02Info(5, 1)    = Omega / (1 - Alpha1 - Alpha2);
ARCH02Info(6, 1)    = sqrt(DaysPerYear) * sqrt(ARCH02Info(5, 1));  
ARCH02Info(7:9, 1)  = ParmTStat;
ARCH02Info(10:12, 1) = ParmPVal;
   
% Then, estimate ARCH(10)
EstimatedModel      = estimate(ARCH10Model, ARCHData);
ARCH10Info(1, 1)    = EstimatedModel.Constant;
for j=1:10
    ARCH10Info(j+1, 1) = EstimatedModel.ARCH{j};
end
ARCH10Info(12, 1)    = sum(ARCH10Info(2:11, 1));
ARCH10Info(13, 1)    = EstimatedModel.UnconditionalVariance;
ARCH10Info(14, 1)    = sqrt(DaysPerYear) * sqrt(EstimatedModel.UnconditionalVariance);     


% Producing displays; for ARCH(1)
RowHeaders              = {'Constant', 'ARCH(1)', 'Variance', 'Unconditional Vol. (Ann.)'};
ColHeaders              = [{' ', 'CVX'}];
DisplayAux              = [RowHeaders' num2cell(ARCH01Info)];
ARCH01Display           = [ColHeaders; DisplayAux];

% Producing display for ARCH(2)
RowHeaders              = {'Constant', 'ARCH(1)', 'ARCH(2)', 'Sum of Alphas', ...
                          'Variance', 'Unconditional Vol. (Ann.)', ...
                          'T-Statistics', '', '', 'P-Values', '', ''};
ColHeaders              = [{' ', 'CVX'}];
DisplayAux              = [RowHeaders' num2cell(ARCH02Info)];
ARCH02Display           = [ColHeaders; DisplayAux];

% Producing display for ARCH(10)
RowHeaders              = {'Constant', 'ARCH(1)', 'ARCH(2)', 'ARCH(3)', 'ARCH(4)', ...
                          'ARCH(5)', 'ARCH(6)', 'ARCH(7)', 'ARCH(8)', 'ARCH(9)', ...
                          'ARCH(10)', 'Sum of Coefficients', 'Variance', ...
                          'Unconditional Vol. (Ann.)'};
ColHeaders              = [{' ', 'CVX'}];
DisplayAux              = [RowHeaders' num2cell(ARCH10Info)];
ARCH10Display           = [ColHeaders; DisplayAux];

% With the coefficient estimates, we can compute in-sample variance for
% the models.  
% Start with ARCH(1)
ARCH01Var               = NaN(NDatesForReturns, 1);

   % First, determine the effective sample for each security
   RowsAux             = find(~isnan(LogReturns(:, 4)));
   StartRowForStock    = min(RowsAux); 
    
   ARCH01Var(StartRowForStock, 1) = ARCH01Info(3, 1);
      
   for j=(StartRowForStock + 1):NDatesForReturns
       ARCH01Var(j, 1)  = ARCH01Info(1, 1) + ...
                           ARCH01Info(2, 1) .* LogReturnsSq(j - 1, 4);
   end


% Finally, convert to volatility
ARCH01Vol                = sqrt(ARCH01Var);

% Then, compute the conditional variances for ARCH(2).  We perform the
% computation only if there are 10 past returns available for it.  The
% following code illustrates an alternative way to test whether enough data
% exists for the computation.

ARCH02Var               = NaN(NDatesForReturns, 1);

   % Extracting the coefficients we stored previously; note that the
   % computation below requires the coefficients to be used in the opposite
   % order we stored them, so we reverse them.
   Const                = ARCH02Info(1, 1);
   ARCHCoeffs           = ARCH02Info(3:(-1):2,1)';
   
   for j=11:NDatesForReturns
       Sample           = (j-2):(j-1);
       if (sum(~isnan(LogReturnsSq(Sample, 4))) == 2)
           ARCH02Var(j, 1)  = Const + ARCHCoeffs * LogReturnsSq(Sample, 4);
       end
   end

% Finally, convert to volatility
ARCH02Vol               = sqrt(ARCH02Var);

% Then, compute the conditional variances for ARCH(10)  
ARCH10Var               = NaN(NDatesForReturns, 1);

   % Extracting the coefficients we stored previously; note that the
   % computation below requires the coefficients to be used in the opposite
   % order we stored them, so we reverse them.
   Const                = ARCH10Info(1, 1);
   ARCHCoeffs           = ARCH10Info(11:(-1):2,1)';
   
   for j=11:NDatesForReturns
       Sample           = (j-10):(j-1);
       if (sum(~isnan(LogReturnsSq(Sample, 4))) == 10)
           ARCH10Var(j, 1)  = Const + ARCHCoeffs * LogReturnsSq(Sample, 4);
       end
   end

% Finally, convert to volatility
ARCH10Vol               = sqrt(ARCH10Var);

%% *** Part 10: Estimate a GARCH(1, 1) model for CVX return series ***

% Create a model first
GARCHModel             = garch(1, 1);

% Estimate GARCH for each series, save information for computing estimates
GARCHInfo              = NaN(6, 1);

   GARCHData           = LogReturns(:, 4);
   GARCHData           = GARCHData(~isnan(GARCHData));
   EstimatedModel      = estimate(GARCHModel, GARCHData);
   GARCHInfo(1, 1)     = EstimatedModel.Constant;
   GARCHInfo(2, 1)     = EstimatedModel.ARCH{1};
   GARCHInfo(3, 1)     = EstimatedModel.GARCH{1};
   GARCHInfo(4, 1)     = sum(GARCHInfo(2:3, 1));
   GARCHInfo(5, 1)     = EstimatedModel.UnconditionalVariance;
   GARCHInfo(6, 1)     = sqrt(DaysPerYear) * sqrt(GARCHInfo(5, 1));

% Producing matrix to display results
RowHeaders             = {'Constant', 'ARCH(1)', 'GARCH(1)', 'Sum of Coeffs.', 'Variance', 'Unconditional Vol. (Ann.)'};
ColHeaders             = [{' ', 'CVX'}];
DisplayAux             = [RowHeaders' num2cell(GARCHInfo)];
GARCHDisplay           = [ColHeaders; DisplayAux];

% With the coefficient estimates, we can compute in-sample GARCH variances
GARCHVar               = NaN(NDatesForReturns, 1);

   GARCHData          = LogReturns(:, 4);
   RowsToUse          = find(~isnan(GARCHData));
   StartRow           = min(RowsToUse);
   
   GARCHVar(StartRow, 1) = GARCHInfo(5, 1);
   
   for j=(StartRow+1):NDatesForReturns
       GARCHVar(j, 1)  = GARCHInfo(1, 1) + ...
                           GARCHInfo(2, 1) .* LogReturnsSq(j - 1, 4) + ...
                           GARCHInfo(3, 1) .* GARCHVar(j - 1, 1);
   end

% Finally, convert to volatility
GARCHVol               = sqrt(GARCHVar);

%% *** Part 11: Produce a chart with log returns and +/- 2 Std. Dev. for models ***

% Choose security to chart (CVX)
i                       = 4;                    % Working with CVX
CVXReturns              = LogReturns(:, i);

% (1) Produce chart for EWMA model
EWMAVolForChart         = EWMAVol(:, 1);

% Put together chart data, remove rows with NaNs, compute start and end year
ChartData               = [CVXReturns 2*EWMAVolForChart -2*EWMAVolForChart];
RowsToUse               = find(~isnan(ChartData(:, 2)));
StartDate               = DatesForReturns(min(RowsToUse));
StartYear               = (StartDate - mod(StartDate, 10000))/10000;
EndDate                 = DatesForReturns(max(RowsToUse));
EndYear                 = (EndDate - mod(EndDate, 10000))/10000;
ChartData               = ChartData(RowsToUse, :);
NChartDates             = size(ChartData, 1);

% Chart options
TimeLabels              = linspace(StartYear, EndYear, NChartDates);
YTickPoints             = linspace(-0.25, 0.25, 11);

% Charting the series
figure(FigNo);
titleStr = strcat({'Daily Returns and +/- 2 Std. Dev. EWMA(0.97) Vol. for '}, Tickers(1, i));
title(titleStr);
xlabel('Dates');
axis([StartYear EndYear -0.25 0.25]);
grid on;
hold on;
plot(TimeLabels, ChartData(:, 1), 'LineStyle', '-' , ...
                  'LineWidth', 1, 'Color', 'blue');
plot(TimeLabels, ChartData(:, 2), 'LineStyle', '--', ...
                 'LineWidth', 1, 'Color', 'red'); 
plot(TimeLabels, ChartData(:, 3), 'LineStyle', '--', ...
                 'LineWidth', 1, 'Color', 'red');
set(gca, 'YTickMode', 'manual');
set(gca, 'YTick', YTickPoints);
set(gca, 'YTickLabel', num2str(100 .* get(gca, 'YTick')', '%1.0f%%'));              
legend({'Log Returns', '2*EWMA(0.97) Vol'}, ...
                     'location','best');            
set(gcf, 'color', 'white');                  
FigNo   = FigNo + 1;

% (3) Produce chart for ARCH(1) model
ARCH01VolForChart       = ARCH01Vol(:, 1);

% Put together chart data, remove rows with NaNs, compute start and end year
ChartData               = [CVXReturns 2*ARCH01VolForChart -2*ARCH01VolForChart];
RowsToUse               = find(~isnan(ChartData(:, 2)));
StartDate               = DatesForReturns(min(RowsToUse));
StartYear               = (StartDate - mod(StartDate, 10000))/10000;
EndDate                 = DatesForReturns(max(RowsToUse));
EndYear                 = (EndDate - mod(EndDate, 10000))/10000;
ChartData               = ChartData(RowsToUse, :);
NChartDates             = size(ChartData, 1);

% Chart options
TimeLabels              = linspace(StartYear, EndYear, NChartDates);
YTickPoints             = linspace(-0.25, 0.25, 11);

% Charting the series
figure(FigNo);
titleStr = strcat({'Daily Returns and +/- 2 Std. Dev. ARCH(1) Vol. for '}, Tickers(1, i));
title(titleStr);
xlabel('Dates');
axis([StartYear EndYear -0.25 0.25]);
grid on;
hold on;
plot(TimeLabels, ChartData(:, 1), 'LineStyle', '-' , ...
                  'LineWidth', 1, 'Color', 'blue');
plot(TimeLabels, ChartData(:, 2), 'LineStyle', '--', ...
                 'LineWidth', 1, 'Color', 'red'); 
plot(TimeLabels, ChartData(:, 3), 'LineStyle', '--', ...
                 'LineWidth', 1, 'Color', 'red');
set(gca, 'YTickMode', 'manual');
set(gca, 'YTick', YTickPoints);
set(gca, 'YTickLabel', num2str(100 .* get(gca, 'YTick')', '%1.0f%%'));              
legend({'Log Returns', '2*ARCH(1) Vol'}, ...
                     'location','best');            
set(gcf, 'color', 'white');                  
FigNo   = FigNo + 1;

% (3) Produce chart for ARCH(2) model
ARCH02VolForChart       = ARCH02Vol(:, 1);

% Put together chart data, remove rows with NaNs, compute start and end year
ChartData               = [CVXReturns 2*ARCH02VolForChart -2*ARCH02VolForChart];
RowsToUse               = find(~isnan(ChartData(:, 2)));
StartDate               = DatesForReturns(min(RowsToUse));
StartYear               = (StartDate - mod(StartDate, 10000))/10000;
EndDate                 = DatesForReturns(max(RowsToUse));
EndYear                 = (EndDate - mod(EndDate, 10000))/10000;
ChartData               = ChartData(RowsToUse, :);
NChartDates             = size(ChartData, 1);

% Chart options
TimeLabels              = linspace(StartYear, EndYear, NChartDates);
YTickPoints             = linspace(-0.25, 0.25, 11);

% Charting the series
figure(FigNo);
titleStr = strcat({'Daily Returns and +/- 2 Std. Dev. ARCH(2) Vol. for '}, Tickers(1, i));
title(titleStr);
xlabel('Dates');
axis([StartYear EndYear -0.25 0.25]);
grid on;
hold on;
plot(TimeLabels, ChartData(:, 1), 'LineStyle', '-' , ...
                  'LineWidth', 1, 'Color', 'blue');
plot(TimeLabels, ChartData(:, 2), 'LineStyle', '--', ...
                 'LineWidth', 1, 'Color', 'red'); 
plot(TimeLabels, ChartData(:, 3), 'LineStyle', '--', ...
                 'LineWidth', 1, 'Color', 'red');
set(gca, 'YTickMode', 'manual');
set(gca, 'YTick', YTickPoints);
set(gca, 'YTickLabel', num2str(100 .* get(gca, 'YTick')', '%1.0f%%'));              
legend({'Log Returns', '2*ARCH(2) Vol'}, ...
                     'location','best');            
set(gcf, 'color', 'white');                  
FigNo   = FigNo + 1;

% (4) Produce chart for ARCH(10) model
ARCH10VolForChart       = ARCH10Vol(:, 1);

% Put together chart data, remove rows with NaNs, compute start and end year
ChartData               = [CVXReturns 2*ARCH10VolForChart -2*ARCH10VolForChart];
RowsToUse               = find(~isnan(ChartData(:, 2)));
StartDate               = DatesForReturns(min(RowsToUse));
StartYear               = (StartDate - mod(StartDate, 10000))/10000;
EndDate                 = DatesForReturns(max(RowsToUse));
EndYear                 = (EndDate - mod(EndDate, 10000))/10000;
ChartData               = ChartData(RowsToUse, :);
NChartDates             = size(ChartData, 1);

% Chart options
TimeLabels              = linspace(StartYear, EndYear, NChartDates);
YTickPoints             = linspace(-0.25, 0.25, 11);

% Charting the series
figure(FigNo);
titleStr = strcat({'Daily Returns and +/- 2 Std. Dev. ARCH(10) Vol. for '}, Tickers(1, i));
title(titleStr);
xlabel('Dates');
axis([StartYear EndYear -0.25 0.25]);
grid on;
hold on;
plot(TimeLabels, ChartData(:, 1), 'LineStyle', '-' , ...
                  'LineWidth', 1, 'Color', 'blue');
plot(TimeLabels, ChartData(:, 2), 'LineStyle', '--', ...
                 'LineWidth', 1, 'Color', 'red'); 
plot(TimeLabels, ChartData(:, 3), 'LineStyle', '--', ...
                 'LineWidth', 1, 'Color', 'red');
set(gca, 'YTickMode', 'manual');
set(gca, 'YTick', YTickPoints);
set(gca, 'YTickLabel', num2str(100 .* get(gca, 'YTick')', '%1.0f%%'));              
legend({'Log Returns', '2*ARCH(10) Vol'}, ...
                     'location','best');            
set(gcf, 'color', 'white');                  
FigNo   = FigNo + 1;

% (5) Produce chart for GARCH(1, 1) model
GARCHVolForChart        = GARCHVol(:, 1);

% Put together chart data, remove rows with NaNs, compute start and end year
ChartData               = [CVXReturns 2*GARCHVolForChart -2*GARCHVolForChart];
RowsToUse               = find(~isnan(ChartData(:, 2)));
StartDate               = DatesForReturns(min(RowsToUse));
StartYear               = (StartDate - mod(StartDate, 10000))/10000;
EndDate                 = DatesForReturns(max(RowsToUse));
EndYear                 = (EndDate - mod(EndDate, 10000))/10000;
ChartData               = ChartData(RowsToUse, :);
NChartDates             = size(ChartData, 1);

% Chart options
TimeLabels              = linspace(StartYear, EndYear, NChartDates);
YTickPoints             = linspace(-0.25, 0.25, 11);

% Charting the series
figure(FigNo);
titleStr = strcat({'Daily Returns and +/- 2 Std. Dev. GARCH(1, 1) Vol. for '}, Tickers(1, i));
title(titleStr);
xlabel('Dates');
axis([StartYear EndYear -0.25 0.25]);
grid on;
hold on;
plot(TimeLabels, ChartData(:, 1), 'LineStyle', '-' , ...
                  'LineWidth', 1, 'Color', 'blue');
plot(TimeLabels, ChartData(:, 2), 'LineStyle', '--', ...
                 'LineWidth', 1, 'Color', 'red'); 
plot(TimeLabels, ChartData(:, 3), 'LineStyle', '--', ...
                 'LineWidth', 1, 'Color', 'red');
set(gca, 'YTickMode', 'manual');
set(gca, 'YTick', YTickPoints);
set(gca, 'YTickLabel', num2str(100 .* get(gca, 'YTick')', '%1.0f%%'));              
legend({'Log Returns', '2*GARCH(1, 1) Vol'}, ...
                     'location','best');            
set(gcf, 'color', 'white');                  
FigNo   = FigNo + 1;

%% *** Part 12 - Estimating various versions of GARCH for CVX ***
CVXReturns             = LogReturns(:, 4);
CVXReturnsSq           = CVXReturns .* CVXReturns;
ParamEstimates         = NaN(6, 4);
TStats                 = NaN(5, 4);

% First, estimate GARCH(1, 1) as a baseline
[Parameters, LogL, ~, VCov] = tarch(CVXReturns, 1 , 0, 1); %1 arch lag, 1 garch lag, 0 leverage lag

ParamEstimates(1, 1)   = Parameters(1, 1); % Omega
ParamEstimates(2, 1)   = Parameters(2, 1); % ARCH coefficient
ParamEstimates(3, 1)   = Parameters(3, 1); % GARCH coefficient

VCovGARCH              = VCov;
LogLGARCH              = LogL;

ParmStdErrorGARCH      = sqrt(diag(VCovGARCH));
ParmTStatsGARCH        = Parameters ./ ParmStdErrorGARCH;

TStats(1, 1)           = ParmTStatsGARCH(1, 1); % Omega
TStats(2, 1)           = ParmTStatsGARCH(2, 1); % ARCH coefficient
TStats(3, 1)           = ParmTStatsGARCH(3, 1); % GARCH coefficient

% Then, estimate GJR-GARCH(1, 1, 1) %with leverage effect
%choose this model in this case
[Parameters, LogL, ~, VCov] = tarch(CVXReturns, 1 , 1, 1); 

ParamEstimates(1, 2)   = Parameters(1, 1);  % Omega
ParamEstimates(2, 2)   = Parameters(2, 1);  % ARCH coefficient
ParamEstimates(3, 2)   = Parameters(4, 1);  % GARCH coefficient
ParamEstimates(4, 2)   = Parameters(3, 1);  % Leverage coefficient

VCovGJRGARCH           = VCov;
LogLGJRGARCH           = LogL;

ParmStdErrorGJRGARCH   = sqrt(diag(VCovGJRGARCH));
ParmTStatsGJRGARCH     = Parameters ./ ParmStdErrorGJRGARCH; %both the parameter and test statistics(significant grater than 2) shows that there is leverage effect

TStats(1, 2)           = ParmTStatsGJRGARCH(1, 1);  % Omega
TStats(2, 2)           = ParmTStatsGJRGARCH(2, 1);  % ARCH coefficient
TStats(3, 2)           = ParmTStatsGJRGARCH(4, 1);  % GARCH coefficient
TStats(4, 2)           = ParmTStatsGJRGARCH(3, 1);  % Leverage coefficient

% Then, estimate GARCH(1, 1) with t-distributed residuals 
% mu need to be very high for the residual to be normal
% need to see if residuals are in fact normal, which will be done later
[Parameters, LogL, ~, VCov] = tarch(CVXReturns, 1 , 0, 1, 'STUDENTST'); 

ParamEstimates(1, 3)   = Parameters(1, 1);  % Omega
ParamEstimates(2, 3)   = Parameters(2, 1);  % ARCH coefficient
ParamEstimates(3, 3)   = Parameters(3, 1);  % GARCH coefficient
ParamEstimates(6, 3)   = Parameters(4, 1);  % Degrees of freedom in t

VCovGARCHt             = VCov;
LogLGARCHt             = LogL;

% Note: not producing t-statistic for nu
ParmStdErrorGARCHt     = sqrt(diag(VCovGARCHt));
ParmTStatsGARCHt       = Parameters(1:(end-1), :) ./ ParmStdErrorGARCHt(1:(end-1), :);

TStats(1, 3)           = ParmTStatsGARCHt(1, 1);  % Omega
TStats(2, 3)           = ParmTStatsGARCHt(2, 1);  % ARCH coefficient
TStats(3, 3)           = ParmTStatsGARCHt(3, 1);  % GARCH coefficient

% Then, estimate Power GARCH; note that the specification used in the
% MFE Toolbox is different from the one in lecture notes
% test statistic is not significant in this case
[Parameters, LogL, ~, VCov] = aparch(CVXReturns, 1 , 0, 1); 

ParamEstimates(1, 4)   = Parameters(1, 1);  % Omega
ParamEstimates(2, 4)   = Parameters(2, 1);  % ARCH coefficient
ParamEstimates(3, 4)   = Parameters(3, 1);  % GARCH coefficient
ParamEstimates(5, 4)   = Parameters(4, 1);  % Power parameter

VCovPGARCH             = VCov;
LogLPGARCH             = LogL;

ParmStdErrorPwrGARCH   = sqrt(diag(VCovPGARCH));
ParmTStatsPwrGARCH     = Parameters(1:(end-1), :) ./ ParmStdErrorPwrGARCH(1:(end-1), :); %test statistics for power pearamter tested separately

% Note: testing whether parameter for power equals 2, instead of zero
TStats(1, 4)           = ParmTStatsPwrGARCH(1, 1);  % Omega
TStats(2, 4)           = ParmTStatsPwrGARCH(2, 1);  % ARCH coefficient
TStats(3, 4)           = ParmTStatsPwrGARCH(3, 1);  % GARCH coefficient
TStats(5, 4)           = (Parameters(4, 1) - 2) / ParmStdErrorPwrGARCH(end, 1);  % Power parameter

% Produce display with results
Models                 = {'GARCH(1, 1)', 'GJR-GARCH', 'GARCH with t Res.', 'Power GARCH'};
RowHeaders             = {'Constant', 'ARCH(1)', 'GARCH(1)', 'Leverage', 'Power', 'Nu' ...
                          't (Const)', 't (ARCH)', 't (GARCH)', 't (Lever.)', 't (Power)'};
ColHeaders             = [{' '} Models];
DisplayAux             = [RowHeaders' [num2cell(ParamEstimates); num2cell(TStats)]];
ResultsDisplay         = [ColHeaders; DisplayAux];

%% *** Part 13 - Coefficient tests and likelihood ratio tests ***
% Parameter tests were carried out in the previous section

% We perform three likelihood-ratio tests.  First, test GJR-GARCH against GARCH
TestStatistic1          = -2 * (LogLGARCH - LogLGJRGARCH);
PValue1                 = (1 - chi2cdf(TestStatistic1, 1));

% Next, test Power GARCH against GARCH
TestStatistic2          = -2 * (LogLGARCH - LogLPGARCH);
PValue2                 = (1 - chi2cdf(TestStatistic2, 1));

% Next, test GARCH(1, 1) with t-distributed residuals against GARCH
TestStatistic3          = -2 * (LogLGARCH - LogLGARCHt);
PValue3                 = (1 - chi2cdf(TestStatistic3, 1));

%% *** Part 14 - Computing standardized residuals, testing them ***

% First, compute GARCH residuals for CVX
CVXGARCHVar            = NaN(NDatesForReturns, 1);
CVXGARCHVar(1, 1)      = ParamEstimates(1, 1) / (1 - ParamEstimates(2, 1) ...
                            - ParamEstimates(3, 1));
                        
for i=2:NDatesForReturns                     
    CVXGARCHVar(i, 1)  = ParamEstimates(1, 1) + ...
                           ParamEstimates(2, 1) * CVXReturnsSq(i-1, 1) + ...
                           ParamEstimates(3, 1) * CVXGARCHVar(i-1, 1);
end

CVXGARCHVol            = sqrt(CVXGARCHVar);
CVXGARCHResid          = CVXReturns ./ CVXGARCHVol;

% Then, compute GARCH-t residuals for CVX
CVXGARCHtVar           = NaN(NDatesForReturns, 1);
CVXGARCHtVar(1, 1)     = ParamEstimates(1, 3) / (1 - ParamEstimates(2, 3) ...
                            - ParamEstimates(3, 3));
                        
for i=2:NDatesForReturns                     
    CVXGARCHtVar(i, 1) = ParamEstimates(1, 3) + ...
                           ParamEstimates(2, 3) * CVXReturnsSq(i-1, 1) + ...
                           ParamEstimates(3, 3) * CVXGARCHtVar(i-1, 1);
end

CVXGARCHtVol           = sqrt(CVXGARCHtVar);
CVXGARCHtResid         = CVXReturns ./ CVXGARCHtVol;

% Produce correlogram for GARCH residuals
%test if the residuals are independent, in this case it is satisfied
figure(FigNo);
autocorr(CVXGARCHResid);
TitleStr               = 'Sample autocorrelation of returns for CVX GARCH Residuals';
title(TitleStr);
ylabel('Correlation');
xlabel('Order');
FigNo                  = FigNo + 1;

% Produce QQ Plot for GARCH residuals
% test if the residuals are normal, in this case it shows it has fatter
% tails than normal distribution
figure(FigNo);
qqplot(CVXGARCHResid);
title('QQ-Plot for CVX GARCH Residuals');
ylabel('CVX GARCH Residuals');
xlabel('Standard normal distribution');
FigNo                  = FigNo + 1;

% Produce correlogram for GARCH-t residuals
% in this case no correlation between residuals
figure(FigNo);
autocorr(CVXGARCHtResid);
TitleStr               = 'Sample autocorrelation of returns for CVX GARCH Residuals';
title(TitleStr);
ylabel('Correlation');
xlabel('Order');
FigNo                  = FigNo + 1;

% Produce QQ Plot for GARCH-t residuals
% Almost linear, but t5 is not really good at capturing what is going on with the 
% residuals on the ends

tdist                 = makedist('tLocationScale', 'mu', 0, ...
                           'sigma',1, 'nu', 8);
figure(FigNo);                       
qqplot(CVXGARCHtResid, tdist);
title('QQ-Plot for CVX GARCH Residuals');
ylabel('CVX GARCH Residuals');
xlabel('t-Distribution with nu = 8');
FigNo                  = FigNo + 1;

