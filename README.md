# Distance.ipynb — Notebook Explanation & README

## Overview

This notebook explores the **Curse of Dimensionality** using geometry, probability, visualization, and machine learning experiments.

The notebook demonstrates:

1. Relationship between a sphere and cube in higher dimensions
2. How distances behave in high-dimensional spaces
3. Why machine learning models become difficult to train with too many features
4. Visualization of geometric intuition using 2D and 3D plots

The overall goal is to build an intuitive understanding of how increasing dimensions affect:

* distance metrics
* data sparsity
* classification accuracy
* geometric volume

---

# Detailed Explanation of Each Section

## 1. Importing Libraries

```python
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```

Additional libraries later include:

```python
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
```

### Purpose

These libraries are used for:

* numerical computation (`numpy`)
* plotting (`matplotlib`)
* machine learning experiments (`scikit-learn`)
* geometry generation (`itertools`, `mplot3d`)

---

# 2. Creating a Circle Visualization

## Function: `make_circle()`

```python
def make_circle(point=0):
```

This function:

* draws a unit circle centered at the origin
* places a point on the x-axis
* visualizes the Euclidean distance from origin

### Concept

The notebook starts from simple 2D geometry:

[
Distance = \sqrt{x^2 + y^2}
]

This creates intuition for distance measurement before moving to higher dimensions.

---

# 3. 3D Cube and Sphere Visualization

The notebook then creates:

* a cube using combinations of corner coordinates
* a sphere using parametric equations

## Cube Generation

```python
for s, e in combinations(np.array(list(product(r,r,r))), 2):
```

This generates all cube edges.

## Sphere Generation

```python
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
```

These equations generate a 3D sphere surface.

### Purpose

This visualization explains:

* how a sphere fits inside a cube
* how geometry changes with dimension
* why volume relationships become important in higher dimensions

---

# 4. Sampling Random Points

```python
sample_data = np.random.sample((5,2))
```

This creates random points inside a 2D space.

## Function: `norm(x)`

This computes the Euclidean norm:

[
||x|| = \sqrt{x_1^2 + x_2^2 + ... + x_n^2}
]

### Meaning

The norm measures:

* distance from origin
* magnitude of vectors
* closeness of points

This becomes extremely important in high-dimensional machine learning.

---

# 5. Percentage of Cube Occupied by Sphere

## Function

```python
what_percent_of_the_ncube_is_in_the_nball()
```

### What it Does

The function:

1. generates random points inside an n-dimensional cube
2. checks whether points lie inside an n-dimensional sphere
3. computes the percentage of points inside the sphere

---

## Key Mathematical Idea

For a point to lie inside the unit sphere:

[
\sum_{i=1}^{n} x_i^2 \leq 1
]

As dimensions increase:

* cube volume grows rapidly
* sphere volume shrinks relative to the cube

### Important Result

In high dimensions:

> Almost all volume of the cube lies outside the sphere.

This is one of the major demonstrations of the **Curse of Dimensionality**.

---

# 6. Plotting Sphere-to-Cube Volume Ratio

```python
plt.plot(dims, data)
```

This graph shows:

* x-axis → dimensions
* y-axis → percentage of cube volume inside sphere

### Interpretation

As dimensions increase:

* percentage approaches zero
* data becomes sparse
* distances lose meaning

This explains why high-dimensional data becomes difficult to analyze.

---

# 7. Distance Concentration Experiment

## Function

```python
def get_min_distance(dimension, sample_size=10**3):
```

### Objective

The notebook studies:

* minimum distance
* maximum distance
* average distance

between random points and the origin.

---

## Important Observation

In higher dimensions:

* nearest points become farther away
* farthest and nearest distances become similar
* distances “concentrate”

This phenomenon is called:

# Distance Concentration

### Why It Matters

Many ML algorithms rely on distance:

* KNN
* clustering
* anomaly detection
* similarity search

When all distances become similar:

* neighbors lose meaning
* models perform poorly

---

# 8. Plotting Distance Statistics

The notebook plots:

* minimum distance
* maximum distance
* mean distance

### Interpretation

As dimensions increase:

* all curves move closer together
* variability in distance decreases
* meaningful geometric intuition breaks down

This is another major effect of high-dimensional spaces.

---

# 9. Machine Learning Classification Experiment

The notebook then demonstrates dimensionality effects using classification.

---

## Dataset Creation

### Two Features

```python
make_classification(n_features=2)
```

### 200 Features

```python
make_classification(n_features=200)
```

Synthetic datasets are generated with different feature counts.

---

# 10. Decision Tree Training

```python
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
```

A Decision Tree classifier is trained and evaluated.

---

# 11. Accuracy Comparison

The notebook compares:

* low-dimensional accuracy
* high-dimensional accuracy

### Main Insight

Adding more features does NOT always improve performance.

In fact:

* irrelevant dimensions add noise
* overfitting increases
* generalization decreases

---

# 12. Increasing Feature Experiment

```python
for num in np.linspace(increment, max_features, increment, dtype='int'):
```

This loop trains models using increasing numbers of features.

The resulting graph shows:

* classification accuracy vs number of features

---

# Final Interpretation

The notebook experimentally proves:

## Curse of Dimensionality

As dimensionality increases:

### Geometry Changes

* volume becomes sparse
* spheres occupy tiny space inside cubes

### Distance Becomes Less Meaningful

* all points become equally distant
* similarity metrics fail

### Machine Learning Suffers

* overfitting increases
* more data is required
* computational complexity grows

---

# Key Concepts Covered

| Concept                 | Meaning                              |
| ----------------------- | ------------------------------------ |
| Euclidean Distance      | Distance between points              |
| Norm                    | Magnitude of vector                  |
| n-ball                  | Higher-dimensional sphere            |
| n-cube                  | Higher-dimensional cube              |
| Distance Concentration  | Distances becoming similar           |
| Curse of Dimensionality | Problems caused by high dimensions   |
| Feature Explosion       | Too many features degrading learning |

---

# Mathematical Concepts Used

## Euclidean Norm

[
||x|| = \sqrt{\sum_{i=1}^{n} x_i^2}
]

## Unit Ball Condition

[
\sum_{i=1}^{n} x_i^2 \leq 1
]

## Sphere Parametric Equations

[
x = \cos(u)\sin(v)
]

[
y = \sin(u)\sin(v)
]

[
z = \cos(v)
]

---

# Real-World Applications

The concepts in this notebook are important in:

* Machine Learning
* Deep Learning
* Recommendation Systems
* Image Processing
* NLP Embeddings
* High-dimensional Statistics
* Clustering Algorithms
* Similarity Search
* Feature Engineering

---


