import requests
from bs4 import BeautifulSoup
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting
import matplotlib.pyplot as plt
import pandas as pd
#from xlwt import Workbook

confirmed = ["SPYG", "QQQ", "USRT", "HYG"]
testing = []
tickers = confirmed + testing
years = 10
latest_year = 2020
rf = 0.01558

#wb = Workbook()
#data = wb.add_sheet('Data')
#col = 1
ret_mat = []
for ticker in tickers:
    # Get the returns from Yahoo Finance
    result = requests.get("https://finance.yahoo.com/quote/" + ticker + "/performance", \
    headers={'User-Agent': 'test-app/1.0'})
    soup = BeautifulSoup(result.content, 'lxml')
    table = soup.find("div", {"data-reactid": "88"})
    returns = [round(float(table.find("span", {"data-reactid": str(i)}).text[:-1]) / 100, 4)\
     for i in range(110, 110 + years * 7, 7)]
    ret_mat.append(returns)
    # Write to the Excel file
    """
    data.write(1, col, ticker)
    for i in range(len(returns)):
        data.write(1 + years - i, col, returns[i])
    col += 1
    """
#wb.save('data.xls')

# Statistics
year_list = [latest_year - i for i in range(years)]
df = pd.DataFrame(ret_mat, tickers, year_list)
exp_r = df.mean(axis=1)
cov = df.transpose().cov() #Covariance, not correlation as in k312

# Efficient frontier
ef1 = EfficientFrontier(exp_r, cov) #Weight bounds default to (0, 1)
fig, ax = plt.subplots() #Figure is the page, axis is the specific graph (default to 1)
plotting.plot_efficient_frontier(ef1, ax=ax, show_assets=True)

# Tangent portfolio
ef2 = EfficientFrontier(exp_r, cov)
w = ef2.max_sharpe(rf)
ret_tang, std_tang, sharpe = ef2.portfolio_performance(True, rf) #first argument is 'verbose'
ax.scatter(std_tang, ret_tang, marker="*", s=100, c="r", label="Tangent Portfolio")

# Display graph with CAL
ax.plot([0, 1.5 * std_tang], [rf, rf + 1.5 * (ret_tang - rf)], label="CAL")
ax.set_title("Efficient Frontier and CAL")
ax.legend()
plt.tight_layout()
plt.show()
print("Tangent Portfolio Weights: ", w)
print(df.transpose().corr())
