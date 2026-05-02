import numpy as np


data = np.load("data/processed/stew_windows.npz")

X = data["X"]
y = data["y"]
groups = data["groups"]

print("X:", X.shape, X.dtype)
print("y:", y.shape, y.dtype)
print("groups:", groups.shape, groups.dtype)

print("Classes:", np.unique(y, return_counts=True))
print("Number of subjects:", len(np.unique(groups)))
print("First 10 groups:", groups[:10])
print("Last 10 groups:", groups[-10:])