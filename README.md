# Machine Learning and Data Science

Practice notebooks from when I was learning the Python data stack: NumPy, Pandas, Matplotlib and scikit-learn. Each folder covers one library, usually with a notebook that teaches the topic and an exercises notebook to drill it.

## Contents

| Folder | Notebooks |
|---|---|
| `Numpy/` | `Numpy.ipynb` (arrays, shapes, indexing, broadcasting, loading images as arrays), `numpy-exercises.ipynb`, `Nut Butter Sales.ipynb` |
| `Pandas/` | `Pandas.ipynb` (Series and DataFrames, CSV in and out, filtering, groupby, missing data), `pandas-exercises.ipynb` |
| `MatPlotLib/` | `MatPlotLib.ipynb`, `MatPlotLib Pandas.ipynb`, `matplotlib-exercises.ipynb`, plus the PNGs they save |
| `Scikit-Learn/` | `Scikit Learn.ipynb` (regression and classification tour, including California Housing and the car-sales data) and `scikit-learn-exercises.ipynb` (longer classification and regression workflows) |
| `Jupyter Notebook/` | `Example.ipynb` (start of a heart disease project) and `Sample.ipynb` (basic notebook) |
| `Data/` | `Car/` (car-sales CSVs, including copies with missing values), `Heart Disease/` (UCI heart disease CSVs), `Images/` (diagrams and sample photos) |

## Running the notebooks

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

The notebooks read data with paths that came from my own machine or from the folder they were first written in: `car-sales.csv`, `heart-disease.csv`, `C:\Miniconda\Project\Data\...` and `../Data/...`. The files are all under `Data/`, but you will need to change the `read_csv` paths in each notebook to point there. `scikit-learn-exercises.ipynb` downloads its car-sales data from a URL instead.

The three full projects that grew out of this are in their own repos: Heart-Disease-Classification, Bulldozer-Price-Prediction and Dog-Breed-Classification.
