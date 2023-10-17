# Future Plans

## PSST (Polymer Solutions Scaling Theory) Library (Python Library)

From the user's perspective:
- Download library from pip or conda
- Read and copy examples on GitHub
- Use library and examples to train models
- Create custom PyTorch modules and train those instead

## PolySolEvaluator (web-app)

From the user's perspective:
- Go to website
- Upload data in .csv format
- Input more data (repeat units molar mass and projection length)
- See numerical results of expected parameters
- See plots of (c vs visc/nw/phi^1.31) (c vs visc/nw/phi^2) (nw/g vs visc), optionally with previous data
- Download .csv files for plots

What happens in the background:
- Data is parsed, normalized, and interpolated onto our grid
- If a significant portion of the data lies outside the grid, train new model on different range of phi/nw (with user's informed consent about time)
- Report expected results, maybe add confidence intervals?
- Plot data and show universality
- Offer the plotted data as individual .csv files
