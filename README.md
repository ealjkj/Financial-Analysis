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

After a grid is trained, we have access to several things. Here we have a few useful methods
```python
grid.values # Is a height x weight matrix that represents the cells of the grid. 
```
