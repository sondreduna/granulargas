---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Event-driven simulation of a granular gas

```python
# importing useful packages

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib import rc
import heapq 

# Setting common plotting parameters

rc("text",usetex = True)
rc("font",family = "sans-serif")

fontsize = 22
newparams = {'axes.titlesize': fontsize,
             'axes.labelsize': fontsize,
             'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 
             'legend.fontsize': fontsize,
             'figure.titlesize': fontsize,
             'legend.handlelength': 1.5, 
             'lines.linewidth': 2,
             'lines.markersize': 7,
             'figure.figsize': (11, 7), 
             'figure.dpi':200,
             'text.usetex' : True,
             'font.family' : "sans-serif"
            }

plt.rcParams.update(newparams)
```

```python
from events import *
```

```python
e1 = Event(1,2,3,4,1)
e2 = Event(0.1,234,234,234,1)
```

```python
e1 < e2
```

```python
collection = Ensemble(100)
```

```python
collection.set_velocities(np.random.random((2,100)))
```

```python
collection.plot_positions()
```

```python
%timeit collection.simulate(100)
```
