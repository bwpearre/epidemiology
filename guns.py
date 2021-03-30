import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy
import matplotlib.pyplot as plt
import itertools
import matplotlib.colors as colors
from pathlib import Path
import seaborn as sns
from pandas.api.types import is_numeric_dtype

def n2f(w):
        w = w.replace('−', '-')     # They look the same, but they're not!
        a = ''.join(itertools.takewhile(lambda x: x in '0123456789-.,eEIiNnFf', w))
        #if a != w:
        #    print(f'Read "{w}", converted to "{a}"')
        try:
            return float(a)
        except:
            return float('NaN')

def sigfig(x, n):
        if x == 0:
                return x
        else:
                return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))

column_rename = {'Rate': 'Homicides (all methods) (per 100,000)',
                 'Rate': 'All gun-related deaths (per 100,000)',
                 'Murders (rate per 100,000 inhabitants)(2010)': 'Homicides (all methods) (per 100,000)',
                 'Homicide': 'Homicides (guns only) (per 100,000)',
                 'Gun murders  (rate per 100,000 inhabitants) (2010)': 'Homicides (guns only) (per 100,000)',
                 'Suicide': 'Suicides, guns only (per 100,000)',
                 'Margin: %': 'Biden vs. Trump: margin (percentage points)',
                 'World Bank Gini[4]: %': 'Income inequality (Gini %)',
                 'Gini Coefficient': 'Income inequality (Gini %)',
                 'Wealth Gini (2019)': 'Wealth inequality (Gini %, 2019)',
                 'Wealth Gini (2018)': 'Wealth inequality (Gini %, 2018)',
                 'Wealth Gini (2008)': 'Wealth inequality (Gini %, 2008)',
                 'Total': 'Total gun-related deaths',
                 }
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
                    'Margin': n2f, # field_converters doesn't work with MultiIndex, but DOES work on the first level
                    }
US_state_index_rename = {'Ala.': 'Alabama',
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


areas = {'USA': {}, 'world': {}}
scaling = 'log'           # ('linear', 'log')

datadir = Path('data')
# Format: filename: (indexcol, headerlines, indexrename, columnrename)
areas['world'] = {'files': {'GunViolence': (0, 0),           # https://en.wikipedia.org/wiki/List_of_countries_by_firearm-related_death_rate
                            'GiniByWealth': (0, 0),          # https://en.wikipedia.org/wiki/List_of_countries_by_wealth_equality
                            'GiniByIncome': (0, [0, 1]),     # https://en.wikipedia.org/wiki/List_of_countries_by_income_equality
                            'Homicides': (0, 0, None, {'Rate': 'Homicides (all methods) (per 100,000)'}),             # https://en.wikipedia.org/wiki/List_of_countries_by_intentional_homicide_rate
                            #'Happiness2020': (1, 0),         # https://en.wikipedia.org/wiki/World_Happiness_Report BUT IT SEEMS TOO SKETCHY
                            'LifeExpectancyWHO': (1, [0,1])  # https://en.wikipedia.org/wiki/List_of_countries_by_life_expectancy
                            }}
areas['USA'] = {'files': {'Homicides': (0, 0),
                          'GunDeaths2013': (0, 0, lambda z: z.split(' (')[0]), # https://en.wikipedia.org/wiki/Firearm_death_rates_in_the_United_States_by_state
                          'GiniByIncome': (1, 0),
                          'Election2020': (0, [0,1], US_state_index_rename)
                          }}

fig = plt.figure(1)
plt.clf()
sp = 0                          # subplot window
# Load everything
for area,a in areas.items():
        df = []
        for f,i in a['files'].items():
                fname = datadir / area / (f + '.csv')
                df.append(pd.read_csv(fname, index_col=i[0], converters=field_converters, header=i[1]))
                for j in df[-1].columns:
                        if type(j) == type(1):
                                # Column name is an int. Probably a ranking.
                                df[-1].drop(j, axis=1, inplace=True)

                if type(i[1]) == type([]):
                        # Having trouble accessing multicolumns. Flatten them.
                        df[-1].columns = [': '.join(col).strip() for col in df[-1].columns.values]

                if len(i) == 3 and i[2] is not None:
                        df[-1].rename(index=i[2], inplace=True)
                if len(i) == 4:
                        df[-1].rename(columns=i[3], inplace=True)
        d = pd.concat(df, axis=1)
        d.rename(columns=column_rename, inplace=True)
        a['d'] = d

        # Gini is represented as 0--1 or 0--100%. Let's force the latter:
        for i in d.columns:
                if "Gini" in i and is_numeric_dtype(d[i]):
                        if np.max(d[i]) <= 1:
                                d[i] = d[i].map(lambda x: x * 100)

        a['d'] = d

        
        
        if area == 'world':
                x = 'CIA Gini[6]'
                x = 'GDP per capita'
                x = 'CIA R/P[5]'
                x = 'UN R/P'
                x = 'Wealth inequality (Gini %, 2018)'
                x = 'Wealth inequality (Gini %, 2019)'
                x = 'Wealth inequality (Gini %, 2008)'
                x = 'Guns per 100 inhabitants'
                x = 'Income inequality (Gini %)'

                y = 'Healthy life expectancy (HALE) at birth: Δ b/w females & males'
                y = 'Healthy life expectancy (HALE) at age 60: Δ b/w females & males'
                y = 'Healthy life expectancy (HALE) at birth: Male'
                y = 'Healthy life expectancy (HALE) at birth: Female'
                y = 'Healthy life expectancy (HALE) at birth: Both'
                y = 'All gun-related deaths (per 100,000)'
                y = 'Guns per 100 inhabitants'
                y = 'Homicides (all methods) (per 100,000)'
                y = 'Suicides, guns only (per 100,000)'
                y = 'Homicides (guns only) (per 100,000)'

                c = 'GDP per capita'

        else:
                x = 'Population density  (inhabitants per square mile) (2010)'
                x = 'Gun ownership (%)'
                x = 'Income inequality (Gini %)'

                y = 'Biden vs. Trump: margin (percentage points)'
                y = 'All gun-related deaths (per 100,000)'
                y = 'Gun ownership (%)(2013)'
                y = 'Homicides (guns only) (per 100,000)'
                y = 'Homicides (all methods) (per 100,000)'


                c = 'Population density  (inhabitants per square mile) (2010)'
                c = 'Gun ownership (%)(2013)'



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
                mask &= (d['GDP per capita'] > 20000)
        #mask &= d.index != 'United States'


        model = LinearRegression().fit(fit_x[mask], fit_y[mask])
        if model.coef_[0] < 0:
                direction = 'in'
        else:
                direction = 'de'
        if scaling == 'log':
                sensitivity = 100*(1-1/10**model.coef_[0])
                print(f'Every 1-point reduction in {x} is associated with a {sigfig(abs(sensitivity), 2)}% {direction}crease in {y}.')
        else:
                sensitivity = model.coef_[0]
                print(f'Every 1-point reduction in {x} is associated with a {sigfig(sensitivity, 2)} {direction}crease in {y}.')   

        fit_test = np.asarray((np.min(d[x]), np.max(d[x]))).reshape(-1,1)

        # Plotting

        sp += 1
        plt.subplot(1, 2, sp)
        #default_size = fig.get_size_inches()
        #fig.set_size_inches( (default_size[0]*2, default_size[1]*2) )
        fig.set_size_inches( 20, 7)
        scat = plt.scatter(x = d[x][mask], y = d[y][mask], c=(d[c][mask]), cmap = cmap, norm = norm)
        plt.colorbar(label=c)
        #scat = plt.scatter(x = d[x][mask], y = d[y][mask])
        if scaling == 'log':
                plt.yscale('log')
                plt.plot(fit_test, 10**model.predict(fit_test))
        else:
                plt.plot(fit_test, model.predict(fit_test))

        plt.title(f'{y} vs. {x}. Slope = {sigfig(sensitivity, 2)}%')
        plt.xlabel(x)
        plt.ylabel(y)
        #plt.rcParams.update({'axes.titlesize': 20, 'axes.labelsize': 20, 'xtick.labelsize': 20})

        for i in d.index[mask]:
            plt.annotate(i, d.loc[i, [x, y]])
        #plt.ylim(bottom = 0.01)

plt.show()

