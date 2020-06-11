# Load base packages:
import pandas as pd
import numpy as np

# Load other key packages
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import calendar

# Stats packages
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Machine learning packages
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
# from xgboost import plot_importance, plot_tree

# Load in the kickstarter finance data
df = pd.read_csv('data/Border_Crossing_Entry_Data.csv')
df.head()
df.tail()

for col in df.columns: 
    print(col) 

# Take a look at the data types
df.dtypes
# Take a look at the descriptive statistics 
df.describe()
# Take a look at the amount of observations in the dataframe
df.shape
df.info()
# Set the date time index
df['Date'] = pd.to_datetime(df['Date'])
df.dtypes

# What is the mean size goal for US border crossings?
df['Value'].mean()

# Set pandas profile report - a useful html compiled exploratory report
from pandas_profiling import ProfileReport
report = ProfileReport(df)

#1. EDA - US Borders:

borders = df['Border'].unique()
print(borders)

# Check which years the the date column extends to
years = df['Date'].map(lambda x : x.year).unique()
years

# Check if the means of arrival column is unique
df['Measure'].unique()

# Clean up value names, and create two new dataframes:
people = df[df['Measure'].isin(['Personal Vehicle Passengers', 'Bus Passengers','Pedestrians', 'Train Passengers'])]
vehicles = df[df['Measure'].isin(['Trucks', 'Rail Containers Full','Truck Containers Empty', 'Rail Containers Empty',
       'Personal Vehicles', 'Buses', 'Truck Containers Full'])]

# Check to see how many have crossed the two borders:
people_borders = people[['Border','Value']].groupby('Border').sum()
people_borders


# Plot the values of US border crossings by border (strip-plot)
sns.set(style="darkgrid")
# Initialize the figure
fg, ax = plt.subplots(dpi = 300)
sns.despine(bottom=True, left=True)
# Show each observation with a strip-plot
sns.stripplot(x="Value", y="Border",
              data=df, dodge=True, alpha=0.2, zorder=1)
# Show the conditional means
sns.pointplot(x="Value", y="Border",
              data=df, dodge=.532, join=False, palette="dark",
              markers="d", scale=.75, ci=None)

# Plot the values of US border crossings by border (pie-chart)
values = people_borders.values.flatten()
labels = people_borders.index
fig = go.Figure(data=[go.Pie(labels = labels, values=values)])
fig.update(layout_title_text='Total inbound persons, since 1996')
fig.show()

# Take the values and set the date as index
p = df[['Date','Border','Value']].set_index('Date')

# Group by years and border
p = p.groupby([p.index.year, 'Border']).sum()
p.head(10)

# Plot total inbound people, by border and years over time
val_MEX = p.loc(axis=0)[:,'US-Mexico Border'].values.flatten().tolist()
val_CAN = p.loc(axis=0)[:,'US-Canada Border'].values.flatten().tolist()
yrs = p.unstack(level=1).index.values

# Bar chart 
fig = go.Figure(go.Bar(x = yrs, y = val_MEX, name='US-Mexico Border', marker_color='rgb(55, 83, 109)'))
fig.add_trace(go.Bar(x = yrs, y = val_CAN, name='US-Canada Border', marker_color='rgb(69,117, 180)'))
fig.update_layout(title = 'Total inbound people, by border and years', barmode='stack', xaxis={'categoryorder':'category ascending'})
fig.show()


# EDA - States and Port Entry Points
# What states are people entering the US in?
df['State'].value_counts()

# What Port's do people enter the US at?
df['Port Name'].value_counts()

# Plot the most popular states to enter the US:
fg, ax = plt.subplots(dpi=300)
sns.stripplot(y='State', x='Value', alpha =0.2, data = df);
ax.set_title('What US state see the most border crossings?')
fg.tight_layout()

# Set the a df that contains the average border crossings for each state
avg_state = df.groupby(df['State'])['Value'].mean()
avg_state = avg_state

# Geospatial Analysis of states:
import plotly.graph_objects as go
import folium 

col = ['Port Name', 'Port Code', 'Border', 'Measure']
map1 = df.drop(col, axis=1)
map1.head()
# Let's create a new dataframe called map1
map1=map1.groupby(["Date","State"])[["Value"]].sum().reset_index()
# US Border Crossing Data by State: Average Data 1996-2020
fig = px.choropleth(map1, locations=map1["State"],       

 color=map1["Value"],
                    locationmode="USA-states",
                    scope="usa",
                    color_continuous_scale='Greens',
                    hover_data = [map1.count],
                    title="US Border Crossings by State: 1996-2020"
                   )

fig.show()

# US Border Crossing Geospatial Data: Month level
cdf=map1
cdf["Date"] = cdf["Date"].astype(str)
fig = px.choropleth(cdf, locations=cdf["State"],       

color=cdf["Value"],locationmode="USA-states", scope="usa",
                    animation_frame=cdf["Date"], color_continuous_scale='Greens', hover_data = [cdf.count],
                    title="US Border Crossings by State: Monthly Data"
                   )

fig.show()

# EDA - Mode of Entry into the US:
#We need to clean this data, as there are some descriptions that do not help us. Let's merge the relevant values.
df_1 = df.replace({"Train Passengers":"Train",
                     "Rail Containers Full":"Train",
                     "Rail Containers Empty":"Train",
                     "Trains":"Train",
                     "Bus Passengers":"Bus",
                     "Buses":"Bus",
                     "Trucks":"Truck",
                     "Truck Containers Full":"Truck",
                     "Truck Containers Empty":"Truck",
                     "Personal Vehicles":"Personal_vehicle",
                     "Personal Vehicle Passengers":"Personal_vehicle"})
df_1.head()

df_1['Measure'].value_counts()

# Plot the mode of entry into the US:
sns.set_style("darkgrid")
sns.set_palette("Paired")
sns.stripplot(y='Measure', x='Value', alpha =0.2, data = df_1);
ax.set_title('How are people crossing the US border?')
fg, ax = plt.subplots(dpi=300)

# Plot the mode of entry into the US by border:
sns.set(style="whitegrid")
# Draw a nested barplot to show survival for class and sex
g = sns.catplot(x="Measure", y="Value", hue="Border", data=df_1,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Value of Border Crossings")

# Similar plot to above:
sns.set_style("darkgrid")
sns.set_palette("Set2")
fg, ax = plt.subplots(dpi=300)
sns.barplot(y='Measure', x='Value', hue="Border", data = df_1);
ax.set_title('How are people crossing the US border?')
ax.set_xscale("log")


import plotly.express as px
ax.set_yscale("log")
fig = px.bar(df_1, x="Measure", y="Value")

fig.update_layout(font=dict(family="Arial",size=10),
    plot_bgcolor='rgb(243,243,243)',paper_bgcolor='rgb(243,243,243)',
    title_text='Box plot distribution for method of transport')
fig.show()

sns.set(style="ticks")
# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(10, 8))
ax.set_xscale("log")
# Plot the orbital period with horizontal boxes
sns.boxplot(x="Value", y="Measure", data=df_1,
            whis=[0, 100], palette="vlag")
# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)


# Take the values and set the date as index
m = people[['Date','Measure','Value']].set_index('Date')
# Group by years and border
m = m.groupby([m.index.year,'Measure']).sum()
m.head(10)


# Bar chart showing the total inbound people by method of arrival and the year of arrival:
measures = ['Personal Vehicle Passengers', 'Bus Passengers','Pedestrians', 'Train Passengers']
yrs = m.unstack().index.values

fig = go.Figure(data = [go.Bar(x = yrs, y = m.loc(axis=0)[:, mes].values.flatten().tolist(), name = mes) for mes in measures ])  

fig.update_layout(title = 'Total inbounds (people), by measure and years', barmode='stack', xaxis={'categoryorder':'category ascending'})
fig.show()

#2: TIME-SERIES ANALYSIS:
# We are going to create a new dataframe and drop the variables that are not needed for this feature analysis
col = ['Port Name', 'State', 'Port Code', 'Border', 'Measure']
df2 = df.drop(col, axis=1)
df2.head()
# Examine the data types
df2.dtypes
# Convert dates from strings to datetime format
df2['Date'] = pd.to_datetime(df2['Date'])
# Set the datetime index
df2 = df2.set_index('Date')
# Check if the index has been set
df2.index


# Time plot of the value of border crossings 
sns.set(rc={'figure.figsize':(10, 8)})
df2['Value'].plot(linewidth=0.8)
# Time plot of the number of border crossings (scatter)
axes = df2['Value'].plot(marker='.', alpha=0.4, linestyle='None', figsize=(11, 9))
axes.set(xlabel="Date",ylabel="Number of People",title="Number of people crossing the US border")

# We are going to create a new dataframe and drop the variables that are not needed for this feature analysis
col = ['Port Name', 'State', 'Port Code','Measure']
df4 = df.drop(col, axis=1)
df4.head()

# Convert dates from strings to datetime format
df4['Date'] = pd.to_datetime(df4['Date'])
# Set the datetime index
df4 = df4.set_index('Date')
# Plot by the total value of border crossings over the years
year = df4.resample('Y').sum()
year['Value'].plot(linewidth=2.5, color="Green")


# Take the values and set the date as index
mb = people[['Date','Border','Measure','Value']].set_index('Date')
# Group by years and border
mb = mb.groupby([mb.index.year,'Border','Measure']).sum()
# Plot only the Mexican border (use the loc function to access the Mexico Border values)
sns.set(rc={'figure.figsize':(15, 8)})
fig,ax = plt.subplots()
mb.loc(axis=0)[:,'US-Mexico Border', :].unstack().Value.plot(title='US-Mexico Border inbound crossings',ax=ax)
fig.tight_layout()
fig.show()
# Plot only the Candian border (use the loc function to access the Canadian Border values)
mb.loc(axis=0)[:,'US-Canada Border', :].unstack().Value.plot(title='US-Canada Border inbound crossings')
plt.show()

# TIME-SERIES FORECASTING:
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Stack the dataframe:
df2 = df2.stack()

from sklearn.model_selection import train_test_split


# Groupby border to examine the relationship between Canada and Mexico
people_crossing_series = people[['Date','Value']].groupby('Date').sum()
people_crossing_series_CAN = people[people['Border'] == 'US-Canada Border'][['Date','Value']].groupby('Date').sum()
people_crossing_series_MEX = people[people['Border'] == 'US-Mexico Border'][['Date','Value']].groupby('Date').sum()

# Plot the total rolling average, Mexico rolling average, Canada rolling average:
sns.set(rc={'figure.figsize':(15, 10)})
fig, ax = plt.subplots(dpi = 300)
# Rolling mean by years
rmean = people_crossing_series.rolling(12, center=True).mean() # Total rolling average
rmean_MEX = people_crossing_series_MEX.rolling(12, center=True).mean() # Mex rolling average
rmean_CAN = people_crossing_series_CAN.rolling(12, center=True).mean() # Canada rolling average

# RA plots:
ax.plot(people_crossing_series, marker='.', linestyle='-', linewidth=1, alpha = 1, label='Total')
ax.plot(rmean,
        marker=None, linestyle='-', linewidth=1.5, alpha = 0.7, label='Total rolling average (years)', color = 'b')

ax.plot(people_crossing_series_MEX, marker='.', linestyle='-', linewidth=1, alpha = 1, label='Mexico', color = 'g')
ax.plot(rmean_MEX,
       marker=None, linestyle='-', linewidth=1.5, alpha = 0.7, label='Mexico rolling average (years)', color = 'g')

ax.plot(people_crossing_series_CAN, marker='.', linestyle='-', linewidth=1, alpha = 1, label='Canada', color = 'r')
ax.plot(rmean_CAN,
       marker=None, linestyle='-', linewidth=1.5, alpha = 0.7, label='Canada rolling average (years)', color = 'r')

ax.set(title = 'Monthly total of people entering the US by Border, from 1996', xlabel = 'year')
ax.legend()
plt.show() # show the plot
plt.savefig('Average Monthly Total of US Border Crossings, from 1996.png', dpi=300)


# Plot the expanding average total vs the rolling average total:
sns.set(rc={'figure.figsize':(15, 10)})
fig, ax = plt.subplots(dpi = 300)
# Rolling mean by years
rmean = people_crossing_series.rolling(12, center=True).mean() # Total rolling average
emean = people_crossing_series.expanding(12, center=True).mean() # Total expanding average

ax.plot(people_crossing_series, marker='.', linestyle='-', linewidth=1, alpha = 1, label='Total')
ax.plot(rmean,
        marker=None, linestyle='-', linewidth=1.5, alpha = 0.7, label='Total rolling average (years)', color = 'b')


ax.plot(people_crossing_series, marker='.', linestyle='-', linewidth=1, alpha = 1, label='Total')
ax.plot(emean,
        marker=None, linestyle='-', linewidth=1.5, alpha = 0.7, label='Total expanding average (years)', color = 'g')


ax.set(title = 'Monthly total of people entering the US by Border, from 1996', xlabel = 'year')
ax.legend()
plt.show() # show the plot
plt.savefig('Average vs Expanding Average Monthly Total of US Border Crossings, from 1996.png', dpi=300)


# Why was there a drop in 2002, that lasted until 2012? Is there are granular relationship here? 
fig, ax = plt.subplots()
start = '2012'
end = '2020'

ax.plot(people_crossing_series.loc[start:end],
       marker='o', linestyle='-', linewidth=0.8, alpha = 1, label='Total', color = 'b')
ax.plot(rmean.loc[start:end],
       marker=None, linestyle='-', linewidth=1.5, alpha = 0.7, label='Total, rolling mean (years)', color = 'b')

ax.plot(people_crossing_series_MEX.loc[start:end],
       marker='.', linestyle='-', linewidth=0.8, alpha = 0.9, label='Mexico', color = 'r')
ax.plot(rmean_MEX.loc[start:end],
       marker=None, linestyle='-', linewidth=1.5, alpha = 0.7, label='Mexico, rolling mean (years)', color = 'g')

ax.plot(people_crossing_series_CAN.loc[start:end],
       marker='.', linestyle='-', linewidth=0.8, alpha = 0.9, label='Canada',color = 'g')
ax.plot(rmean_CAN.loc[start:end],
       marker=None, linestyle='-', linewidth=1.5, alpha = 0.7, label='Canada, rolling mean (years)', color = 'r')

ax.set(title = 'Total people entering the US by Border, from {} to {}'.format(start, end))
ax.legend()
plt.show()

# Normalized averages
import matplotlib.gridspec as mgrid

fig = plt.figure()
grid = mgrid.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])

seas = fig.add_subplot(grid[0])
trend = fig.add_subplot(grid[1], sharex = seas)

start = '2015'
end = '2018'

seas.plot(people_crossing_series.loc[start:end]/people_crossing_series.loc[start:end].sum(),
       marker='o', linestyle='-', linewidth=0.8, alpha = 1, label='Total', color = 'b')

seas.plot(people_crossing_series_MEX.loc[start:end]/people_crossing_series_MEX.loc[start:end].sum(),
       marker='.', linestyle='-', linewidth=0.8, alpha = 0.9, label='Mexico', color = 'r')

seas.plot(people_crossing_series_CAN.loc[start:end]/people_crossing_series_CAN.loc[start:end].sum(),
       marker='.', linestyle='-', linewidth=0.8, alpha = 0.9, label='Canada', color = 'g')

seas.set(title = 'Persons entering in the US, from {} to {}, normalised'.format(start, end),
      ylabel = 'arbitrary units')
seas.legend()

trend.plot(rmean.loc[start:end]/rmean.loc[start:end].sum(),
       marker='', linestyle='-', linewidth=2, alpha = 1, label='Total', color = 'b')

trend.plot(rmean_MEX.loc[start:end]/rmean_MEX.loc[start:end].sum(),
       marker='', linestyle='-', linewidth=2, alpha = 1, label='Mexico', color = 'g')

trend.plot(rmean_CAN.loc[start:end]/rmean_CAN.loc[start:end].sum(),
       marker='', linestyle='-', linewidth=2, alpha = 1, label='Canada', color = 'r')

trend.set(ylabel = ' Trend (arbitrary units)')
fig.tight_layout()
plt.show()



# Plot showing persons entering between 2015 and 2018 by most popular month
start = '2015'
end = '2018'
pcsm = people_crossing_series.loc[start:end]

fig, ax = plt.subplots(2,figsize = (18,13))

for i in range(11) :
    mm = pcsm[pcsm.index.month == i] 
    ax[0].plot(mm, label = calendar.month_abbr[i])
    ax[1].plot(mm/mm.sum(), label = calendar.month_abbr[i])
    
ax[0].set(title = 'persons entering the US between {} and {}, total by months'.format(start, end),
         ylabel = '# people')
ax[1].set(title = 'persons entering the US between {} and {}, trend by months'.format(start, end),
         ylabel = 'arbitrary units')
ax[0].legend()
ax[1].legend()

plt.show()

# Plot showing monthly trends between 2015 and 2018 (normalized)
start = '2011'
end = '2018'
pcsm = people_crossing_series.loc[start:end]
months = [calendar.month_abbr[m] for m in range(1,13)]
fig, ax = plt.subplots(2,figsize = (18,13))

start = int(start)
end = int(end)

for i in range(start, end) :
    yy = pcsm[pcsm.index.year == i];
    yy = yy.set_index(yy.index.month);
    ax[0].plot(yy
               , label = i)
    ax[1].plot(yy/yy.sum()
               , label = i)
    
ax[0].set(title = 'persons entering the US between {} and {}, total by years'.format(start, end),
         ylabel = '# people')

ax[1].set(title = 'persons entering the US between {} and {}, seasonal (normalised)'.format(start, end),
         ylabel = 'arbitrary units')

plt.setp(ax, xticks = range(1,13), xticklabels = months)
ax[0].legend()
plt.tight_layout()
plt.show()



#3: SEASONAL DECOMPOSITION
pcsm = people_crossing_series.loc['2015':]
res_mul = seasonal_decompose(pcsm, model='multiplicative', extrapolate_trend='freq')
res_add = seasonal_decompose(pcsm, model='additive', extrapolate_trend='freq')

# Plot
fig, axes = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(15,8))

res_mul.observed.plot(ax=axes[0,0], legend=False)
axes[0,0].set_ylabel('Observed')

res_mul.trend.plot(ax=axes[1,0], legend=False)
axes[1,0].set_ylabel('Trend')

res_mul.seasonal.plot(ax=axes[2,0], legend=False)
axes[2,0].set_ylabel('Seasonal')

res_mul.resid.plot(ax=axes[3,0], legend=False)
axes[3,0].set_ylabel('Residual')

res_add.observed.plot(ax=axes[0,1], legend=False)
axes[0,1].set_ylabel('Observed')

res_add.trend.plot(ax=axes[1,1], legend=False)
axes[1,1].set_ylabel('Trend')

res_add.seasonal.plot(ax=axes[2,1], legend=False)
axes[2,1].set_ylabel('Seasonal')

res_add.resid.plot(ax=axes[3,1], legend=False)
axes[3,1].set_ylabel('Residual')

axes[0,0].set_title('Multiplicative')
axes[0,1].set_title('Additive')
    
plt.tight_layout()
plt.show()

# Trend vs Residuals
des = res_mul.trend * res_mul.resid
des.plot(figsize = (15,10))
plt.show()


# ADF statistic to test null hypothesis
index_list = des.index
values = list(des)
d = {'Value':values} 
des = pd.DataFrame(d, index = index_list) 
result = adfuller(des.Value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Check for autocorrelation in the time series:
fig, axes = plt.subplots(3, 2, figsize=(16,10))

axes[0, 0].plot(des.Value)
axes[0, 0].set_title('Original Series')
plot_acf(des, ax=axes[0, 1])

axes[1, 0].plot(des.Value.diff()); axes[1, 0].set_title('1st Order Differentiation')
plot_acf(des.diff().dropna(), ax=axes[1, 1])

axes[2, 0].plot(des.diff().diff()); axes[2, 0].set_title('2nd Order Differentiation')
plot_acf(des.diff().diff().dropna(), ax=axes[2, 1])

plt.tight_layout()
plt.show()

# ARIMA Model
model = ARIMA(des, order=(0,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# Plot residuals, density and autocorrelation
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(2,2, figsize=(15,8))
residuals.plot(title="Residuals", ax=ax[0,0])
residuals.plot(kind='kde', title='Density', ax=ax[0,1])
plot_acf(model_fit.resid.dropna(), ax=ax[1,0])
plt.tight_layout()
plt.show()

# Check the forecast vs the actual value 
model_fit.plot_predict()
plt.show()

# Check the forecasted values 
train = des[:74]
test = des[74:]
model_train = ARIMA(train, order=(0,1,1))  

fitted_train = model_train.fit(disp=-1)  
fc, se, conf = fitted_train.forecast(36, alpha=0.05)  # 95% conf
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

plt.figure(figsize=(15,8))
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()