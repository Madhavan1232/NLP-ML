import pandas as pd
import os , sys
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(os.path.join(sys.path[0],input()))

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df = df.dropna(subset=['Close_diff'])

size = int(len(df['Close_diff']) * 0.8)
train = df['Close_diff'][:size]
test = df['Close_diff'][size:]

print(f"Training data size: {len(train)}")
print(f"Testing data size: {len(test)}\n")

best_aic = float("inf")
best_order = None
best_model_type = ""

for p in range(1 , 6):
    model = ARIMA(train, order=(p,0,0))
    model_fit = model.fit()
    aic = model_fit.aic
    print(f"AR({p}) AIC: {aic}")
    if aic < best_aic:
        best_aic = aic
        best_order = (p,0,0)
        best_model_type = f"AR({p})"
for q in range(1 , 6):
    model = ARIMA(train, order=(0,0,q))
    model_fit = model.fit()
    aic = model_fit.aic
    print(f"MA({q}) AIC: {aic}")
    if aic < best_aic:
        best_aic = aic
        best_order = (0,0,q)
        best_model_type = f"MA({q})"

best_model_fit = ARIMA(train, order=best_order).fit()
print(f"Best model: {best_model_type}")

print(best_model_fit.summary())

print("\nLjung-Box Test Results:")
lb_test = acorr_ljungbox(best_model_fit.resid, lags=[1], return_df=True)
print(lb_test)