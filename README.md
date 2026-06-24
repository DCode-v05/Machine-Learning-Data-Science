# Machine Learning & Data Science — Learning Journey

**My working notebooks for the core Python data stack: NumPy, Pandas, Matplotlib, and Scikit-Learn, plus the end-to-end ML workflows I built while learning.**

![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)

## Overview

This is where I started with machine learning. It's the oldest repo on my account (first commit December 2024) and it holds the notebooks I worked through to learn the standard Python data and ML stack from the ground up — array math in NumPy, dataframes and cleaning in Pandas, plotting in Matplotlib, and supervised modelling in Scikit-Learn.

It isn't a single application. It's a structured set of practice notebooks and exercise sets, each focused on one library, plus a couple of full classification and regression workflows that tie everything together. The datasets I trained on (car sales, heart disease) live in the repo so the notebooks run end to end. It also acts as the index to the three project repos I built off the back of this work: heart-disease classification, bulldozer price regression, and dog-breed image classification.

Roughly 171 MB total, all Jupyter Notebook, organised by library so each topic is self-contained.

## Key Features

- **NumPy fundamentals** — arrays, ndarray shapes/dtypes, indexing and slicing, broadcasting, vectorized arithmetic, aggregation, reshaping, and reading an image straight into a NumPy array with `matplotlib.image.imread`.
- **Pandas data handling** — Series and DataFrames, importing/exporting CSVs, selecting and filtering rows, `groupby` aggregation, handling missing values, and column transforms. Includes a separate exercises notebook and a small "Nut Butter Sales" worksheet.
- **Matplotlib visualization** — the pyplot API and the object-oriented (figure/axes) API, line/bar/scatter/histogram plots, plotting directly from Pandas objects, styling, and saving figures to PNG. Anatomy-of-a-plot reference images are included.
- **Scikit-Learn modelling** — both regression and classification, covering the full workflow: train/test split, preprocessing, fitting, prediction, and evaluation.
- **Two complete ML workflows** — an end-to-end classification pipeline and an end-to-end regression pipeline, each with model comparison, hyperparameter tuning, cross-validation, and metric reporting.
- **Model persistence** — exporting a trained model to disk with `joblib` and loading it back.
- **Bundled datasets** — car-sales CSVs (clean, extended, and with deliberately missing data for imputation practice) and the UCI heart-disease dataset, so notebooks are runnable as-is.
- **Project index** — links out to the three standalone repos built from these foundations.

## How It Works

The repo is split by library. Each folder has a "teach" notebook and usually a matching exercises notebook, so I'd learn a concept and then immediately drill it.

### Data foundations — NumPy, Pandas, Matplotlib

`Numpy/Numpy.ipynb` (58 cells) walks through ndarray creation, dtypes and shapes, indexing, broadcasting, and vectorized operations, ending with loading image files (`car-photo.png`, `dog-photo.png`, `panda.png`) into arrays to show that an image is just a 3D NumPy array. `numpy-exercises.ipynb` is the practice set.

`Pandas/Pandas.ipynb` (65 cells) covers Series/DataFrame construction, reading and writing CSVs (it exports a transformed car-sales file to `Export_Car.csv`), filtering, `groupby`, and missing-data handling. `pandas-exercises.ipynb` repeats the patterns as drills.

`MatPlotLib/` has three notebooks: `MatPlotLib.ipynb` for the core pyplot/OO plotting API, `MatPlotLib Pandas.ipynb` for plotting straight from DataFrames, and a larger `matplotlib-exercises.ipynb`. Sample outputs are saved as PNGs alongside.

### Modelling — Scikit-Learn

`Scikit-Learn/Scikit Learn.ipynb` (53 cells) is the guided tour and uses real toy datasets:

- **Regression** on California Housing (`fetch_california_housing`) and on the car-sales data. It one-hot-encodes categorical columns with `OneHotEncoder` + `ColumnTransformer`, splits with `train_test_split`, then fits `RandomForestRegressor` and `Ridge`, and scores with `.score()` and `mean_absolute_error`.
- **Classification** on Iris and Olivetti Faces. It compares `LinearSVC` and `RandomForestClassifier`, runs `cross_val_score`, and evaluates with a confusion matrix (`ConfusionMatrixDisplay`), `classification_report`, ROC curve (`roc_curve`), and `roc_auc_score`.

### The end-to-end workflows

`Scikit-Learn/scikit-learn-exercises.ipynb` (107 cells) is the most complete piece. It runs two full pipelines:

1. **Classification workflow** — get data ready → pick a model → fit and predict → evaluate. It compares four estimators (`LinearSVC`, `SVC`, `LogisticRegression`, `RandomForestClassifier`) by collecting their test scores into a dictionary, then tunes `LogisticRegression` with `RandomizedSearchCV`, refits with the best params, and reports precision/recall/F1, confusion matrix, ROC, and cross-validated accuracy. It finishes by exporting the trained model with `joblib.dump` and loading it back with `joblib.load`.

2. **Regression workflow** — builds a proper preprocessing `Pipeline` with `ColumnTransformer`, using `SimpleImputer` (median for numeric, constant fill for categorical/door columns) and `OneHotEncoder(handle_unknown="ignore")`, then chains it to a model step. It compares `Ridge`, `SVR`, and `RandomForestRegressor` inside the pipeline and reports `mean_absolute_error`, `mean_squared_error`, and `r2_score`.

### Starter project notebook

`Jupyter Notebook/Example.ipynb` sets up the Heart Disease project the standard way — problem definition, data, and a 6-step ML framework reference image — which is the template I then expanded into the full classification repo.

## Results / Highlights

The notebooks themselves are practice work, but the projects they led to have concrete numbers:

- **Heart-Disease-Classification** — UCI binary classification (LogisticRegression, SVM, KNN, RF, and others with grid/randomized search), best around **85% accuracy**.
- **Bulldozer-Price-Prediction** — Kaggle "Bluebook for Bulldozers" regression with `RandomForestRegressor` + `RandomizedSearchCV`, **RMSLE ≈ 0.255**.
- **Dog-Breed-Classification** — image classification with a CNN in TensorFlow/Keras.

## Tech Stack

- **Language:** Python 3.x
- **Notebooks:** Jupyter
- **Data / numerics:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **ML:** Scikit-Learn (ensemble, linear_model, svm, neighbors, model_selection, preprocessing, compose, impute, pipeline, metrics), joblib for model persistence
- **Deep learning (in linked projects):** TensorFlow / Keras

## Getting Started

### Prerequisites
- Python 3.x
- pip
- Jupyter (or VS Code / JupyterLab)

### Installation
```bash
git clone https://github.com/DCode-v05/Machine-Learning-Data-Science.git
cd Machine-Learning-Data-Science
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Running
```bash
jupyter notebook
```
Then open any notebook and run the cells top to bottom.

## Usage

Pick a folder based on what you want to look at:

- Start in `Numpy/`, then `Pandas/`, then `MatPlotLib/` for the data foundations.
- Move to `Scikit-Learn/Scikit Learn.ipynb` for the guided modelling tour.
- Open `Scikit-Learn/scikit-learn-exercises.ipynb` for the two full classification and regression pipelines.

The CSVs in `Data/` (and the duplicates next to the notebooks that need them) make every notebook runnable without downloading anything. The `*-missing-data.csv` car files are intentionally incomplete so you can practise imputation.

## Project Structure

```
Machine-Learning-Data-Science/
├── Numpy/
│   ├── Numpy.ipynb              # arrays, broadcasting, vectorized ops, image-as-array
│   ├── numpy-exercises.ipynb    # NumPy practice set
│   ├── Nut Butter Sales.ipynb   # small NumPy/Pandas worksheet
│   └── numpy-images/            # sample images loaded into arrays
├── Pandas/
│   ├── Pandas.ipynb             # Series/DataFrames, CSV I/O, groupby, missing data
│   ├── pandas-exercises.ipynb   # Pandas practice set
│   └── Export_Car.csv           # exported transform output
├── MatPlotLib/
│   ├── MatPlotLib.ipynb         # pyplot + object-oriented plotting API
│   ├── MatPlotLib Pandas.ipynb  # plotting directly from DataFrames
│   ├── matplotlib-exercises.ipynb
│   └── *.png                    # saved sample figures
├── Scikit-Learn/
│   ├── Scikit Learn.ipynb       # regression + classification guided tour
│   └── scikit-learn-exercises.ipynb  # full classification & regression pipelines
├── Jupyter Notebook/
│   ├── Example.ipynb            # Heart Disease project starter (6-step framework)
│   └── Sample.ipynb             # minimal notebook
├── Data/
│   ├── Car/                     # car-sales CSVs (clean, extended, missing-data variants)
│   ├── Heart Disease/           # UCI heart-disease datasets
│   └── Images/                  # reference diagrams + sample images
└── README.md
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
