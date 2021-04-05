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
import re
import ast
import numbers

# Convert a string to a float---or at least as much as possible before the string goes stringy (e.g. "123.45 [in Aughust '84]" --> 123.45)
def n2f(w):
        w = w.replace('−', '-')     # They look the same, but they're not!
        a = ''.join(itertools.takewhile(lambda x: x in '0123456789-.,eEIiNnFf', w))
        #if a != w:
        #    print(f'Read "{w}", converted to "{a}"')
        try:
            return float(a)
        except:
            return float('NaN')

# Return a string representation of x with n significant figures
def sigfig(x, n):
        if x == 0:
                return "0"
        else:
                return str(round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1)))

# "data/*/files.csv" contains special fields for cleaning our raw data. In particular,
# "Index rename" and "Column rename" can be NaN or Dict or "read up to 'str'". Convert the latter two to what they really are, safely
def parse_file_converters(string):
        # What converter functions do we recognise?
        legal_converter_functions = {'n2f'}
        
        if type(string) == str and len(string) > 0:
                if 'readupto' in string:
                        # Provide a string like " (" after which input will be discarded; e.g. "New Hampshire (N.H.)"
                        upto = re.findall('"([^"]*)"', string)[0] # Get the first quoted string after the keyword:
                        return lambda z: z.split(upto)[0] # Split on the split string, return first element
                else:
                        # Provide a Dictionary of converters. Check if the converter is a legal function, and if so, point to it.
                        a = ast.literal_eval(string)
                        if type(a) == dict:
                                for k,v in a.items():
                                        if v in legal_converter_functions:
                                                # Convert the name of a legal converter function to the actual function:
                                                a[k] = globals()[v]
                                                
                        return a
        # Apparently I can just not bother to return a new value.



areas = {'USA': {}, 'world': {}}
scaling = 'log'           # ('linear', 'log')

datadir = Path('data')

fig = plt.figure(1)
plt.clf()
sp = 0                          # subplot window


# Load everything
for area,a in areas.items():
        # Read the file that describes the actual data files:
        files = pd.read_csv(datadir / area / 'files.csv', sep='\t', index_col='Filename', converters={'Index rename': parse_file_converters, 'Column rename': parse_file_converters, 'Clean': parse_file_converters})
        
        for c in files.columns:
                if c == 'Header lines':
                        # Header lines is numeric but maybe a list; e.g. 1 or [1,2,5]. It will have been read as a string. Convert it here:
                        files[c] = files[c].map(lambda z: ast.literal_eval(z))
                        
        df = []
        for f in files.index:
                fname = datadir / area / (f + '.csv')
                df.append(pd.read_csv(fname, index_col=files.loc[f,'Index column'], header=files.loc[f,'Header lines'], converters=files.loc[f,'Clean']))

                # Column name is an int. Probably a ranking. Kill it,
                # because I can't figure out a better way. This has to
                # be done before the multiindex check below...
                for j in df[-1].columns:
                        if isinstance(j, numbers.Number):
                                df[-1].drop(j, axis=1, inplace=True)

                # MultiIndices create annoying special cases. Flatten the names.
                if type(files.loc[f,'Header lines']) == list:
                        df[-1].columns = [': '.join(col).strip() for col in df[-1].columns.values]

                # Rename indices, if requested
                if type(files.loc[f,"Index rename"]) == dict or callable(files.loc[f,"Index rename"]):
                        df[-1].rename(index=files.loc[f,"Index rename"], inplace=True)

                # Rename columns, if requested
                if type(files.loc[f,"Column rename"]) == dict:
                        df[-1].rename(columns=files.loc[f,"Column rename"], inplace=True)
                        
        d = pd.concat(df, axis=1)

        print(f'***** Possible indices for the {area} *****\n{d.dtypes}')

        # Gini is represented as 0--1 or 0--100%. Let's force the latter:
        for i in d.columns:
                if "Gini" in i and is_numeric_dtype(d[i]):
                        if np.max(d[i]) <= 1:
                                d[i] = d[i] * 100
        if area == 'world':
                x = 'CIA Gini[6]'
                x = 'GDP per capita'
                x = 'UN R/P'
                x = 'Wealth inequality (Gini %, 2018)'
                x = 'Wealth inequality (Gini %, 2019)'
                x = 'Wealth inequality (Gini %, 2008)'
                x = 'Guns per 100 inhabitants'
                x = 'Income inequality (Gini %)'
                x = 'CIA R/P[5]: 10%'
                d[x] = d[x].map(lambda x: np.log10(x))

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
                y = 'Murders (all methods) (per 100,000)'


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

        plt.title(f'{y.split("(")[0]} vs. {x.split("(")[0]} --  Slope = {sigfig(sensitivity, 2)}%')
        plt.xlabel(x)
        plt.ylabel(y)
        #plt.rcParams.update({'axes.titlesize': 20, 'axes.labelsize': 20, 'xtick.labelsize': 20})

        for i in d.index[mask]:
            plt.annotate(i, d.loc[i, [x, y]])
        #plt.ylim(bottom = 0.01)

plt.show()

