import numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy.stats as st

btc = pd.read_csv(
        "btc_1y.csv",
        skiprows=3,                               # sla Price- & Ticker-regels plus lege Date-regel over
        names=["Date","Open","High","Low","Close","Adj Close","Volume"],
        index_col="Date",
        parse_dates=["Date"]
     )

ret = np.log(btc["Close"]).diff().dropna()

# statistieken
desc = ret.describe()
skew, kurt = ret.skew(), ret.kurtosis()
sharpe = ret.mean() / ret.std()
jb, pval = st.jarque_bera(ret)

print(desc.to_string())
print(f"Skew = {skew: .3f},  Kurtosis = {kurt: .3f}")
print(f"Sharpe = {sharpe: .3f}")
print(f"Jarque-Bera χ² = {jb: .2f},  p = {pval: .4f}")

# histogram en QQ-plot
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.hist(ret, bins=50, alpha=0.7, edgecolor="k")
plt.title("BTC log-returns (histogram)")
plt.subplot(1,2,2)
st.probplot(ret, dist="norm", plot=plt)
plt.title("QQ-plot vs Normal")
plt.tight_layout()
plt.savefig("../btc_returns_stats.png", dpi=150)
print("Plot saved → ../btc_returns_stats.png")
