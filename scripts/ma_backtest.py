"""
Simple moving-average crossover strategy:
• Long when MA20 > MA50, flat otherwise
• Uses daily BTC-USD data from btc_1y.csv
• Compares strategy CAGR & Sharpe to Buy-and-Hold
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, os

btc = pd.read_csv("btc_1y.csv", skiprows=3,
                  names=["Date","Open","High","Low","Close","Adj Close","Volume"],
                  index_col="Date", parse_dates=["Date"])

price = btc["Close"]

# -------- M A   C A L C S ----------------------------------
ma20 = price.rolling(20).mean()
ma50 = price.rolling(50).mean()

signal = (ma20 > ma50).astype(int)          # 1 = long, 0 = flat
ret     = np.log(price).diff()
strat_r = ret * signal.shift(1)             # enter at next bar

# -------- P E R F O R M A N C E ----------------------------
def annual_cagr(r):
    return np.exp(r.sum())**(365/len(r)) - 1

def sharpe(r):
    return r.mean() / r.std() * np.sqrt(365)

out = {
    "CAGR_strat" : annual_cagr(strat_r.dropna()),
    "CAGR_buy"   : annual_cagr(ret.dropna()),
    "Sharpe_strat": sharpe(strat_r.dropna()),
    "Sharpe_buy"  : sharpe(ret.dropna()),
}
print(out)

# -------- E Q U I T Y  C U R V E ---------------------------
eq_buy = ret.cumsum().apply(np.exp)
eq_str = strat_r.cumsum().apply(np.exp)

plt.plot(eq_buy, label="BuyHold")
plt.plot(eq_str, label="MA20>50")
plt.title("Equity curve")
plt.legend(); plt.tight_layout()
plt.savefig("../ma_equity.png", dpi=150)
print("Plot → ma_equity.png")
