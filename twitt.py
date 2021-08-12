import matplotlib.pyplot as plt
import numpy as np

covid = [539, 3475, 6213, 8758, 8237]

covid_cum = []

for week in covid:
    ind = covid.index(week)
    if ind == 0:
        covid_cum.append(week)
    else:
        med = week + covid_cum[ind-1]
        covid_cum.append(med)

covid_announced = (759, 3605, 8958, 14576, 19506)


dt_cov = (11141, 16387, 18516, 22351, 21997)
dt_ave = (10130, 10305, 10520, 10497, 10458)

excess = []
ct = 0
for wk in dt_cov:
    hold = dt_cov[ct] - dt_ave[ct]
    excess.append(hold)
    ct+=1

print(excess)
h = 0
for i in excess:
    excess[h] = excess[h] * 1.14
    h+=1

excess_cum = []
for w in excess:
    indi = excess.index(w)
    if indi == 0:
        excess_cum.append(w)
    else:
        medium = w + excess_cum[indi-1]
        excess_cum.append(medium)


ln = len(covid_cum)

x = ['Mar 27', 'Apr 3', 'Apr 10', 'Apr 17', 'Apr 24']

ratio = []
cw = 0
for ww in covid_cum:
    helper = excess_cum[cw] - covid_announced[cw]
    hlp = (100/covid_announced[cw]) * helper
    ratio.append(hlp)
    cw+=1
print(ratio)


# print(covid_announced)
# print(excess_cum)
# print(ratio)

plt.subplot(2, 1, 1)
plt.plot(x, excess_cum)
plt.plot(x, covid_announced)
plt.subplot(2, 1, 2)
plt.plot(x, ratio)
plt.show()