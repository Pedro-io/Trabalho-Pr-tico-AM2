from sklearn import decomposition
from sklearn import datasets
from babyplots import Babyplot
import numpy as np

np.random.seed(5)

# load the data set
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create the babyplots visualization
bp = Babyplot()
bp.add_plot(X.tolist(), "shapeCloud", "categories", y.tolist(), {"shape": "sphere",
                                                                 "colorScale": "Set2",
                                                                 "showAxes": [True, True, True],
                                                                 "axisLabels": ["PC 1", "PC 2", "PC 3"]})
# show the visualization
bp