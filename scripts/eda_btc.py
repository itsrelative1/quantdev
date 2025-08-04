import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

btc = yf.download("BTC-USD", period="1y", interval="1d")

print("Rows:", len(btc))
print(btc.describe())

btc["MA20"] = btc["Close"].rolling(20).mean()
btc["MA50"] = btc["Close"].rolling(50).mean()

plt.style.use("ggplot")
btc[["Close", "MA20", "MA50"]].plot(figsize=(10, 5), title="BTC-USD 1Y")
plt.ylabel("Price (USD)")
plt.tight_layout()
plt.savefig("../btc_1y_ma.png", dpi=150)
print("Plot saved as btc_1y_ma.png")

btc.to_csv("../btc_1y.csv")