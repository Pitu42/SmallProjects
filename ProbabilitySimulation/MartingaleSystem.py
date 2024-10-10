import numpy as np
import matplotlib.pyplot as plt

stockList = []
stock = 100000
startingAmount = 10
actualAmount = startingAmount
win = True
index = 0
while stock - actualAmount*2 > 0:
	if win == True:
		actualAmount = startingAmount
	elif win == False:
		actualAmount *= 2
	randomNumber = np.random.randint(0,2)
	if np.random.randint(0,2) == 0:
		stock -= actualAmount
		win = False
	else:
		stock += actualAmount
		win = True
	stockList.append(stock)
print(stockList)
plt.plot(stockList)
plt.xlabel("Tries")
plt.ylabel("Money")
plt.show()
