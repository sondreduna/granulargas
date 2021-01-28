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
import seaborn as sns

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
cm = plt.get_cmap("viridis")
```

```python
plt.scatter(1,1, color = cm(40))
```

```python
v_1 = np.random.random((2,10000))
v_2 = np.random.random((2,10000))
```

```python
a = np.array([1,1,1,1,10,101,10])

print(np.argmin(a))
```

```python
v_1 = np.array([[1,2,3,4,5],[2,4,6,8,10]])
v_2 = np.array([[2,2,2,2,2],[2,2,2,2,2]])
```

```python
v = np.array([1,1])
```

```python
v_1 + np.reshape(v,(2,1))
```

```python
%timeit v_1 * v_2
```

```python
%timeit np.einsum('ij,ij->i',v_1,v_2)
```

```python
%timeit np.multiply(v_1,v_2)
```

```python
%timeit np.sum(v_1 * v_2, axis = 0)
```

```python
np.einsum('ij,ij->j',v_1,v_2)
```

```python
T = np.full(10000,np.inf)
```

```python
T[ (v_1 * v_2 < 0.5) * (v_1 > 0.5) ] = 1
```

```python
from events import *
```

```python
collection = Ensemble(2)
```

```python
collection.radii = np.array([0.01, 0.3])
collection.M     = np.array([1, 10])
collection.particles[:2,0] = np.array([[0.2,0.6]])
collection.particles[:2,1] = np.array([[0.6,0.5]])
```

```python
collection.set_velocities([[1,0.],[0,0.]])
```

```python
collection.plot_positions()
```

```python
collection.simulate(10, True)
```

```python
collection = Ensemble(4)

collection.M = np.array([1,1,1,1]) * 10
collection.particles[:2,0] = np.array([[0.2,0.5]])
collection.particles[:2,1] = np.array([[0.6,0.5]])
collection.particles[2:,0] = np.array([[1,0]])
collection.particles[:2,2] = np.array([[0.1,0.1]])
collection.particles[2:,2] = np.array([1,1])
```

```python
collection.plot_positions()
```

```python
collection.simulate(1000,True)
```

```python
N = 50
```

```python
collection = Ensemble(N, 0.01)
```

```python
v_0 = 5

theta = np.random.random(N) * np.pi * 2

v = v_0 * np.array([np.cos(theta),np.sin(theta)])
collection.set_velocities(v)
```

```python
collection.plot_positions()
```

```python
collection.plot_velocity_distribution(r"\textbf{Initial distribution}")
```

```python
collection.simulate(100,True)
```

```python
collection.plot_velocity_distribution(r"\textbf{Final distribution}",compare = True, savefig = "../fig/vel_dist.pdf")
```

```python
collection.particles[:2,:]
```

```python

```
