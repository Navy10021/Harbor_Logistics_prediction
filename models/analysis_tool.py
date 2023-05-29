import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"
plot_template = dict(
    layout=go.Layout({
        "font_size": 15,
        "xaxis_title_font_size": 15,
        "yaxis_title_font_size": 15})
)
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.model_selection import train_test_split


# 1. Load meta dataset
df = pd.read_csv("./data/meta_data(cargo).csv", index_col = "date")


# 2. Visualization of meta dataset
fig = px.line(df, labels = dict(created_at="date", variable=""))
fig.update_layout(template = plot_template,
                  legend = dict(orientation = 'h', y = 1.2, title_text = " "))
fig.show()


# 3. Trend analysis
print("\n >> Analyze Harbor logistics Trend...")

# Plot Graph format
plt.rc("figure", autolayout=True, 
       figsize=(15, 6), titlesize = 18, titleweight = 'bold')
plt.rc("axes", labelweight = "bold", labelsize="large",
       titleweight = "bold", titlesize = 16, titlepad = 10)
plot_params = dict(color = '0.75', style = ".-",
                   markeredgecolor = "0.25", markerfacecolor = "0.25", legend = False)
%config InlineBackend.figure_format = 'retina'

# Define the predictor variable as Busan Port(Korea's No. 1 export/import port). The user can select a different port.
target = "Busan"  

# One year(12 months) moving average
MA = df[target].rolling(window = 12, center = True).mean()
ax = df[target].plot(style = ".", color = "0.5")
MA.plot(ax=ax, linewidth = 3, title = "12-Month Moving Average of Port Traffic Volumes (Busan)", legend = False)

# Extract Busan port's trend data from DeterministicProcess
dp = DeterministicProcess(
    index = df[target].index,  # dates from the training data
    constant = True,           # dummy feature for the bias (y_intercept)
    order = 1,                 # the time dummy (Trend)
    drop = True                # drop terms if necessary to avoid collinearity
    )              

X = dp.in_sample()
y = df[target]

# Linear Regression Model to predict port traffic volumes trend
model = LinearRegression(fit_intercept = False)
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index = X.index)

ax = df[target].plot(style=".", color="0.5", title = "Port Traffic Volumes (Busan) - Linear Trend")
_ = y_pred.plot(ax=ax, linewidth=3, label = "Trend")
plt.show()

# 12 Months forcast from Trend
X_fore = dp.out_of_sample(steps = 12)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = df[target].plot(title = "Forcasting Port Traffic Volumes Next 12 Months (Busan) - Linear Trend", **plot_params)
ax = y_pred.plot(ax=ax, linewidth = 3, label="Trend")
ax = y_fore.plot(ax=ax, linewidth = 1, label="Trend Forecast",
                 linestyle="-", color='#e35f62', marker='*', markersize = 5)
_ = ax.legend()
plt.show()


# 4. Seasonality analysis
print("\n >> Analyze Harbor logistics Seasonality...")

# date index to datetime
df.index = pd.to_datetime(df.index, utc = False)

# Create seasonality function
def build_seasonality(dataframe):
    dataframe = pd.DataFrame(dataframe)
    dataframe['Quarter'] = dataframe.index.quarter
    dataframe['Month'] = dataframe.index.month
    dataframe['Year'] = dataframe.index.year

    return dataframe

df_season = build_seasonality(df[target])
#df_season.head()

# Seasonality by Quarter
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data = df_season, x = 'Quarter', y = target, palette = 'Blues')
ax.set_title('Seasonality of Port Traffic Volumn by Quarter')
plt.show()

# Seasonality by Month
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data = df_season, x = 'Month', y = target, palette = 'Blues')
ax.set_title('Seasonality of Port Traffic Volumn by Month')
plt.show()


# 5. Time Series analysis
print("\n >> Analyze Harbor logistics Time Series...")

# Build lag data function : Lag 1 = Delay One-Year
def make_lag(df):
    df['Lag_1'] = df[target].shift(1)
    df['Lag_3'] = df[target].shift(3)
    df['Lag_6'] = df[target].shift(6)
    df['Lag_9'] = df[target].shift(9)
    df['Lag_12'] = df[target].shift(12)
    df = df.reindex(columns=[target, 'Lag_1','Lag_3','Lag_6','Lag_9','Lag_12','Time'])

# Build lag data function_2
def make_target_lag(dataframe, lags):
    return pd.concat({f'y_lag_{i}':dataframe.shift(i) for i in range(1, lags+1)}, axis = 1)

target = "Busan"
target_df = pd.DataFrame(df[target])
target_df['time'] = np.arange(len(target_df))
make_lag(target_df)

# Serial Dependence Visualization
plt.subplot(1, 2, 1)
ax = sns.regplot(x = 'Lag_1', y = target, data = target_df,
                ci = None, scatter_kws = dict(color='0.25'))
ax.set_aspect('equal')
ax.set_title('Lag One Month Port Traffic Volumes')

plt.subplot(1, 2, 2)
ax = sns.regplot(x = 'Lag_3', y = target, data = target_df,
                ci = None, scatter_kws = dict(color='0.25'))
ax.set_aspect('equal')
ax.set_title('Lag Tree Months Port Traffic Volumes')

# Plot Graph format
plt.rc("figure", autolayout=True, 
       figsize=(15, 6), titlesize = 18, titleweight = 'bold')
plt.rc("axes", labelweight = "bold", labelsize="large",
       titleweight = "bold", titlesize = 16, titlepad = 10)
plot_params = dict(color = '0.75', style = ".-",
                   markeredgecolor = "0.25", markerfacecolor = "0.25")
%config InlineBackend.figure_format = 'retina'

def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax

def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig
  
  # Choose the best lags : Lag 1 ~ 2 months
target_df.index = pd.to_datetime(target_df.index, utc = False)
X = make_target_lag(target_df.Busan, lags = 2)
X = X.fillna(0.0)

# Linear Regression Model to predict port traffic volumes series
y = target_df.Busan.copy()
# Test Set : Forcast 12 Months
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 12, shuffle=False)

# Train & prediction
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = pd.Series(model.predict(X_train), index = y_train.index)
y_fore = pd.Series(model.predict(X_val), index = y_val.index)

# Visualization
ax = y_train.plot(**plot_params)
ax = y_val.plot(**plot_params)
ax = y_fore.plot(ax=ax, label="Series Forcast", linestyle="-", color='#e35f62', marker='*', markersize = 5)
ax = y_pred.plot(ax=ax, label="Fitted", color='C0')
_ = ax.legend()
plt.title("Forcasting Port Traffic Volumes Next 12 Months (Busan) - Linear Series", pad = 20)
plt.show()

ax = y_val.plot(**plot_params)
_ = y_fore.plot(ax=ax, color='#e35f62', label="Forcast",
               linestyle="--", marker='*')
ax.legend();

mae_valid = metrics.mean_absolute_error(y_val, y_fore)
rmse_valid = np.sqrt(metrics.mean_absolute_error(y_val, y_fore))
rmsle_valid = metrics.mean_squared_log_error(y_val, y_fore) ** 0.5
print(" >> RMSE Score : {:.4f}".format(rmse_valid))
print(" >> RMSLE Score : {:.4f}".format(rmsle_valid))


print("\n >> Harbor Logistics Data Analysis is done.")
