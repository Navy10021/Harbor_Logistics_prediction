import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

# 1. Load meta dataset
df = pd.read_csv("./data/meta_data(cargo).csv", index_col = "date")


# 2. Define the predictor variable as Busan Port(Korea's No. 1 export/import port). The user can select a different port.
target = "Busan"


# 3. Features selection based with Pearson Correlation Coefficient(PCC) : PCC > +0.55 and PCC < -0.55
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap = "Blues")
plt.title("Feature Importance (Correlations Between Variables)", size = 16)
plt.show()
PCC_score = 0.55
num_cols = list(df.corr()[target][(df.corr()[target] > PCC_score) | (df.corr()[target]< -PCC_score)].index)
new_df = df[num_cols]
print("\n >> Important features based on PCC :", list(new_df.columns))


# 4. Feature selection with Ensemble model (XGBoost)
train = df.loc[df.index < '2020-01-01']
test = df.loc[df.index >= '2020-01-01']
features = list(df.columns)
features.remove(target)

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

reg = xgb.XGBRegressor(base_score = 0.5, booster = 'gbtree',    
                       n_estimators = 6000,
                       early_stopping_rounds = 50,
                       objective = 'reg:linear',
                       max_depth = 3,
                       learning_rate = 0.001)
reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose = 1000)

plt.figure(figsize=(10, 8))
fi = pd.DataFrame(
    data=reg.feature_importances_,
    index=reg.feature_names_in_,
    columns=['importance'])
fi.sort_values('importance').plot(kind = 'barh')
plt.title("Feature Importance (Ensemble Model)", size = 16)
plt.legend(loc = 'center right')
plt.show()


print("Feature selection is Done.")
