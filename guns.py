import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy
import matplotlib.pyplot as plt
import itertools
import matplotlib.colors as colors
from pathlib import Path
from pandas.api.types import is_numeric_dtype

def n2f(w):
        w = w.replace('−', '-')     # They look the same, but they're not!
        a = ''.join(itertools.takewhile(lambda x: x in '0123456789-.,eEIiNnFf', w))
#        if a != w:
#            print(f'Read "{w}", converted to "{a}"')
        try:
            return float(a)
        except:
            return float('NaN')

def sigfig(x, n):
        if x == 0:
                return x
        else:
                return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))

field_converters = {'Guns per 100 inhabitants': n2f,
                    'Homicide': n2f,
                    'Suicide': n2f,
                    'Unintentional': n2f,
                    'Undetermined': n2f,
                    'Total': n2f,
                    'World Bank Gini[4]': n2f,
                    'UN R/P': n2f,
                    'CIA Gini[6]': n2f,
                    'Population (in thousands)': n2f,
                    'CIA R/P[5]': n2f,
                    'Gun ownership (%)(2013)': n2f,
                    'Margin': n2f # field_converters doesn't work with MultiIndex, but DOES work on the first level
                    }
US_state_index_converters = {'Ala.': 'Alabama',
                    'Ark.': 'Arkansas',
                    'Calif.': 'California',
                    'Colo.': 'Colorado',
                    'Conn.': 'Connecticut',
                    'Del.': 'Delaware',
                    'D.C.': 'District of Columbia',
                    'Ky.': 'Kentucky',
                    'La.': 'Louisiana',
                    'Md.': 'Maryland',
                    'Maine †': 'Maine',
                    'Mass.': 'Massachusetts',
                    'Miss.': 'Mississippi',
                    'Mich.': 'Michigan',
                    'Minn.': 'Minnesota',
                    'Mo.': 'Missouri',
                    'Mont.': 'Montana',
                    'Neb. †': 'Nebraska',
                    'Nev.[o]': 'Nevada',
                    'N.H.': 'New Hampshire',
                    'N.J.[p]': 'New Jersey',
                    'N.M.': 'New Mexico',
                    'N.Y.[q]': 'New York',
                    'N.C.': 'North Carolina',
                    'N.D.': 'North Dakota',
                    'Okla.': 'Oklahoma',
                    'Pa.': 'Pennsylvania',
                    'R.I.': 'Rhode Island',
                    'S.C.': 'South Carolina',
                    'S.D.': 'South Dakota',
                    'Tenn.': 'Tennessee',
                    'Texas[s]': 'Texas',
                    'Vt.': 'Vermont',
                    'Va.': 'Virginia',
                    'Wash.': 'Washington',
                    'W.Va.': 'West Virginia',
                    'Wis.': 'Wisconsin',
                    'Wyo.': 'Wyoming'}



area = 'world'                                      # ('usa', 'world')
if area == 'world':
        datadir = Path('data/world')
        # Format: filename: (indexcol, headerlines)
        files = {'GunViolence': (0, 0),           # https://en.wikipedia.org/wiki/List_of_countries_by_firearm-related_death_rate
                 'GiniByWealth': (0, 0),          # https://en.wikipedia.org/wiki/List_of_countries_by_wealth_equality
                 'GiniByIncome': (0, [0, 1]),     # https://en.wikipedia.org/wiki/List_of_countries_by_income_equality
                 'Homicides': (0, 0),             # https://en.wikipedia.org/wiki/List_of_countries_by_intentional_homicide_rate
                 #'Happiness2020': (1, 0),         # https://en.wikipedia.org/wiki/World_Happiness_Report BUT IT SEEMS TOO SKETCHY
                 'LifeExpectancyWHO': (1, [0,1])  # https://en.wikipedia.org/wiki/List_of_countries_by_life_expectancy
                 }
else:
        datadir = Path('data/USA')
        files = {'Homicides': (0, 0),
                 'GiniByIncome': (1, 0),
                 'Election2020': (0, [0,1])
                 }

# Load everything
df = []
for f,i in files.items():
        fname = datadir / (f + '.csv')
        #print(f'----- loading {f} -----')
        df.append(pd.read_csv(fname, index_col=i[0], converters=field_converters, header=i[1]))
        for j in df[-1].columns:
                if type(j) == type(1):
                        # Column name is an int. Probably a ranking.
                        df[-1].drop(j, axis=1, inplace=True)
                        
        if type(i[1]) == type([]):
                # Having trouble accessing multicolumns. Flatten them.
                df[-1].columns = [': '.join(col).strip() for col in df[-1].columns.values]
                
        if area == 'usa':
                df[-1].rename(index=US_state_index_converters, inplace = True)
d = pd.concat(df, axis=1)

for i in d.columns:
        if "Gini" in i and is_numeric_dtype(d[i]):
                if np.max(d[i]) <= 1:
                        d[i] = d[i].map(lambda x: x * 100)

if area == 'world':
        x = 'CIA Gini[6]'
        x = 'GDP per capita'
        x = 'CIA R/P[5]'
        x = 'UN R/P'
        x = 'Guns per 100 inhabitants'
        x = 'Wealth Gini (2019)'
        x = 'World Bank Gini[4]: %'

        y = 'Healthy life expectancy (HALE) at birth: Δ b/w females & males'
        y = 'Healthy life expectancy (HALE) at age 60: Δ b/w females & males'
        y = 'Healthy life expectancy (HALE) at birth: Male'
        y = 'Healthy life expectancy (HALE) at birth: Female'
        y = 'Healthy life expectancy (HALE) at birth: Both'
        y = 'Suicide'
        y = 'Rate' # All murders, from https://en.wikipedia.org/wiki/List_of_countries_by_intentional_homicide_rate
        y = 'Homicide'
        
        c = 'GDP per capita'
        c = 'Guns per 100 inhabitants'
        w = 'GDP per capita'
else:
        x = 'Gun ownership (%)(2013)'
        x = 'Population density  (inhabitants per square mile) (2010)'
        x = 'Gini Coefficient'
        
        y = 'Gun murders  (rate per 100,000 inhabitants) (2010)'
        y = 'Margin: %'
        y = 'Murders (rate per 100,000 inhabitants)(2010)'
        
        c = 'Population density  (inhabitants per square mile) (2010)'
        c = 'Gun ownership (%)(2013)'
        w = 'Gini Coefficient'

scaling = 'log'           # ('linear', 'log')

ncolors = 3
boundaries = np.nanquantile((d[c]), np.linspace(0, 1, ncolors+1))
cmap = plt.get_cmap('jet')
norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

fit_x = d[x].to_numpy().reshape(-1,1)
fit_y = d[y].to_numpy()

mask = np.isfinite(d[x]) & np.isfinite(d[y])

if scaling == 'log':
        fit_y = np.log10(fit_y)
        mask &= (d[y] > 0)

if False:
        # Look only at rich countries:
        mask &= (d['GDP per capita'] > np.nanmedian(d['GDP per capita']))


model = LinearRegression().fit(fit_x[mask], fit_y[mask])
if model.coef_[0] < 0:
        direction = 'increase'
else:
        direction = 'decrease'
if scaling == 'log':
        print(f'Every 1-point reduction in {x} is associated with a {sigfig(100*np.abs(1-1/10**model.coef_[0]), 2)}% {direction} in {y}.')
else:
        print(f'Every 1-point reduction in {x} is associated with a {sigfig(np.abs(model.coef_[0]), 2)} {direction} in {y}.')   

fit_test = np.asarray((np.min(d[x]), np.max(d[x]))).reshape(-1,1)

# Plotting


fig = plt.figure()
default_size = fig.get_size_inches()
fig.set_size_inches( (default_size[0]*2, default_size[1]*2) )
scat = plt.scatter(x = d[x][mask], y = d[y][mask], c=(d[c][mask]), cmap = cmap, norm = norm)
plt.colorbar(label=c)
#scat = plt.scatter(x = d[x][mask], y = d[y][mask])
if scaling == 'log':
        plt.yscale('log')
        plt.plot(fit_test, 10**model.predict(fit_test))
else:
        plt.plot(fit_test, model.predict(fit_test))

plt.title('Gun violence vs. Income inequality')
plt.xlabel(x)
plt.ylabel(y)
plt.rcParams.update({'axes.titlesize': 20, 'axes.labelsize': 20, 'xtick.labelsize': 20})

for i in d.index[mask]:
    plt.annotate(i, d.loc[i, [x, y]])
#plt.ylim(bottom = 0.01)
plt.show()

