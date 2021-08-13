# Financial-Analysis
We replicate the ITRAC algorithm in order to forecast the S&amp;P 500 

## Example of use 
Assuming you have a timeseries to train your grid
```python
import numpy as np
ts = np.array([2,5,1,2,3,4,5])

grid = Grid(50,50)
grid.train(ts)
```
