# Financial-Analysis
We replicate the ITRAC algorithm in order to forecast the S&amp;P 500 

## Example of use 
If we have a timeseries we can train a grid. The `trace_size`should be smaller than the lenght of the timeseries.
```python
import numpy as np
ts = np.array([2,5,1,2,3,4,5])

grid = Grid(50,50, trace_size=3)
grid.train(ts)
```

After a grid is trained, we have access to several things. Here we have a few useful attributes
```python
grid.height # Height of the grid
grid.width # Width of the grid
grid.values # Is a height x weight matrix that represents the cells of the grid. 
grid.temp_values # Is a height x weight x 2 tensor. The chanels are the ammount of wins and losses respectively
```

## Grid type
We have two types of grid. The first type is `grid_type='wins'` witch stores the ammount of wins, and losses.
```python
grid = Grid(50,50, grid_type='earnings' trace_size=3)
grid.train(ts)
```
The list of grid types is 
1. **wins:** stores the ammount of wins, and losses. Every win represent +1, and every loss -1.
2. **earnings:** stores the proportion of win or loss. Every win could be 1.4 or 1.1
