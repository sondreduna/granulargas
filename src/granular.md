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
m = 1
M = 25
r = 0.01
R = 0.05

x = np.arange(r,1-r,3*r)
y = np.arange(r,(1-r)/2,3*r)
```

```python
np.size(x)*np.size(y) + 1
```

```python
xx,yy = np.meshgrid(x,y)
```

```python
P = np.zeros((2,np.size(x)*np.size(y) + 1))
```

```python
for i in range(np.size(x)):
```

```python
plt.plot(y,"bo");
```

```python
from events import *
```

```python
collection = Gas(1000,0.0005)
collection.set_velocities(np.random.random((2,1000)))
collection.simulate()
```

```python
theta = np.random.random(1000)*2*np.pi
v = np.array([np.cos(theta),np.sin(theta)])
v2 = np.einsum('ij,ij->j',v,v)

v2[0] *= v2[1]
```

```python
sns.histplot(v2,stat="density")
```

```python
v2 = collection
```

```python
v = np.concatenate([collection1.particles[2:], collection2.particles[2:]],axis = 1)
```

```python
v
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
N = 100
```

```python
collection = Ensemble(N, 0.01)
```

```python
v_0 = 1

theta = np.random.random(N) * np.pi * 2

v = v_0 * np.array([np.cos(theta),np.sin(theta)])
collection.set_velocities(v)
```

```python
collection.plot_positions()
```

```python
%load_ext memory_profiler
```

```python
%mprun collection.simulate(100)
```

```python
collection.plot_velocity_distribution(r"\textbf{Final distribution}",compare = True, savefig = "../fig/vel_dist.pdf")
```

```python
collection.particles[:2,:]
```

```python
v = np.random.random((2,10000))
a = np.random.random(2)
```

```python
b_1 = np.sum(v,)
```

```python
b_2 = v - np.reshape(a,(2,1))
```

```python
b_1 == b_2
```

```python
N = 10
```

```python
from problems import *
```

```python
crater()
```

```python
m = 1
M = 10
r = 0.005
R = 0.05

x = np.arange(3/2 * r,1 - 3/2 * r, 3 * r )
y = np.arange(3/2 * r,(1 - 3/2 * r)/4, 3 * r )

xx,yy = np.meshgrid(x,y)
```

```python
xx
```

```python
yy
```

```python

```
