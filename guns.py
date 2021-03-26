import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy
import matplotlib.pyplot as plt
import itertools

def firstword2float(w):
        a = ''.join(itertools.takewhile(lambda x: x in '0123456789-.,eEIiNnFf', w))
        #if a != w:
        #    print(f'Read "{w}", converted to "{a}"')
        try:
            return float(a)
        except:
            return float('NaN')

def tercile(x, quants):
    if x < quants[0]:
        return 0
    elif x < quants[1]:
        return 1
    elif x < 99999:
        return 2
    else:
        return float('NaN')

field_converters = {'Guns per 100 inhabitants': firstword2float,
                    'Homicide': firstword2float,
                    'Suicide': firstword2float,
                    'Unintentional': firstword2float,
                    'Undetermined': firstword2float,
                    'Total': firstword2float,
                    'World Bank Gini[4]': firstword2float,
                    'UN R/P': firstword2float,
                    'CIA Gini[6]': firstword2float,
                    'Population (in thousands)': firstword2float,
                    'CIA R/P[5]': firstword2float
                    }

files = ('data/GunViolenceByCountry.csv', 'data/wealth.csv', 'data/GiniByIncome.csv', 'data/homicides.csv')
df = []
for f in files:
    df.append(pd.read_csv(f, index_col=0, converters=field_converters))
d = pd.concat(df, axis=1)

x = 'CIA Gini[6]'
x = 'GDP per capita'
x = 'CIA R/P[5]'
x = 'UN R/P'
x = 'Guns per 100 inhabitants'
x = 'Wealth Gini (2019)' # Careful: some of these are 0-100 and some are 0-1
x = 'World Bank Gini[4]'

y = 'Suicide'
y = 'Rate' # All murders, from https://en.wikipedia.org/wiki/List_of_countries_by_intentional_homicide_rate
y = 'Homicide'

quantiles = np.nanquantile(d['GDP per capita'], (.3333, .6666))
for i in d.index:
    d.loc[i, 'tercile'] = tercile(d.loc[i, 'GDP per capita'], quantiles)

fit_x = d[x].to_numpy().reshape(-1,1)
fit_y = d[y].to_numpy()

# Mask out the invalid entries, and also entries we may want to hide for analysis:
mask = np.isfinite(d[x]) & np.isfinite(d[y]) & (d[y] > 0)
if False:
    # Look only at rich countries:
    mask &= (d['GDP per capita'] > np.nanmedian(d['GDP per capita']))


model = LinearRegression().fit(fit_x[mask], np.log10(fit_y[mask]))
print(f'=== Every 1-point reduction in {x} is associated with a {100*(1-1/10**model.coef_[0]):.0f}% reduction in {y}. ===')
fit_test = np.asarray((np.min(d[x]), np.max(d[x]))).reshape(-1,1)

#scat = plt.scatter(x = d[x], y = d[y], s = 10*d['Guns per 100 inhabitants'], c=d['tercile'])
scat = plt.scatter(x = d[x][mask], y = d[y][mask], s = 10*d['Guns per 100 inhabitants'][mask], c=d['GDP per capita'][mask])
plt.plot(fit_test, 10**model.predict(fit_test))
#scat = plt.scatter(x = d[x], y = d['Suicide'], s = 10*d['Guns per 100 inhabitants'], c='blue')
#scat = plt.scatter(x = d[x], y = d[y])
plt.yscale('log')
plt.title('Gun violence vs. Income inequality')
plt.xlabel(x)
plt.ylabel('Homicide per 100,000')
plt.rcParams.update({'axes.titlesize': 20, 'axes.labelsize': 20, 'xtick.labelsize': 20})

for i in d.index[mask]:
    plt.annotate(i, d.loc[i, [x, y]])
#plt.ylim(bottom = 0.01)
plt.show()

