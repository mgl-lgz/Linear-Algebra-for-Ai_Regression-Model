import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

print("---------Part A: Problem and Dataset--------")

df = pd.read_csv("chicken_dataset.csv")
#To remove the rows of any missing values in my dataset. Example year 2018.
df = df.dropna(subset=["Price", "Previous_Month", "Previous_Year"])      
feature_names = ["Previous_Month", "Previous_Year"]
X_raw = df[feature_names].to_numpy()
y = df["Price"].to_numpy()

assert X_raw is not None and y is not None, "Load your dataset above before continuing."
n_samples, n_features = X_raw.shape
print(f"n_samples = {n_samples}, n_features = {n_features}")
print(f"feature_names = {feature_names}")

pd.DataFrame(X_raw, columns=feature_names).describe()

#---------------------------------------------------------
#--------------------------PART B-------------------------
#---------------------------------------------------------

print("\n---------Part B: Matrices-------")

# TODO: Build the design matrix X (rows = samples, columns = features).
# If X_raw is already a clean numeric matrix, this can be as simple as:
X = np.asarray(X_raw, dtype=float)   # TODO: confirm/adjust this is correct for your data
print("X shape:", X.shape)

# TODO: Compute the matrix product X^T X
XtX = X.T@X   # TODO: replace with the correct matrix product
print("X^T X shape:", None if XtX is None else XtX.shape)

# Inspect one off-diagonal entry of X^T X (pick any i != j)
i, j = 0, 1   # TODO: choose two different feature indices
print(f"(X^T X)[{i},{j}] = {XtX[i, j]:.4f}  -- corresponds to features "
f"'{feature_names[i]}' and '{feature_names[j]}'")

#---------------------------------------------------------
#--------------------------PART C-------------------------
#---------------------------------------------------------

print("\n---------Part C: Vector Spaces-------")
# TODO: Compute the rank of your design matrix X
rank_X = np.linalg.matrix_rank(X)   # TODO: use a NumPy linear-algebra function
n_samples, n_features = X.shape
print(f"X is {n_samples} x {n_features}.  rank(X) = {rank_X}")

# TODO: Compare rank(X) to the number of features.
# Are the columns of X linearly independent? Print a short conclusion.

if rank_X == n_features:
    print("The columns of X are linearly independent.")
else:
    print("The columns of X are linearly dependent.")


#---------------------------------------------------------
#--------------------------PART D-------------------------
#---------------------------------------------------------

print("\n---------Part D: Linear Transformation-------")
# TODO: Build the matrix representation of your chosen linear transformation.
# Example: standardization T(x) = S(x - mean), whose LINEAR part is
# multiplication by S = diag(1/std):
mu = X.mean(axis=0)
sigma = X.std(axis=0)
S = np.diag(1 / sigma)   # TODO: build the diagonal matrix S = diag(1/sigma)

#For standard deviations, mean and such
print("Mean:", mu)
print("Standard Deviation:", sigma)
print("Transformation Matrix:\n", S)
print("Determinant of S:", np.linalg.det(S))

X_centered = X - mu
X_transformed = X_centered @ S   # TODO: apply T(x) = S x  (i.e. X_centered @ S)
print("X_transformed shape:", X_transformed.shape)

# Show the transformation applied to ONE real data point
idx = 2  # TODO: pick a sample index
x_before = X[idx]
x_after = X_transformed[idx] if X_transformed is not None else None
print("Before:", x_before)
print("After: ", x_after)

# TODO: Split your (transformed) data into train and test sets.
# Example:
dates = pd.to_datetime(df["Date"], format="%m/%d/%Y")
train_idx = (dates < "2025-01-01").to_numpy()
test_idx = ~train_idx

mu = X[train_idx].mean(axis=0)
sigma = X[train_idx].std(axis=0)

X_train = (X[train_idx] - mu) / sigma
X_test = (X[test_idx] - mu) / sigma
y_train = y[train_idx]
y_test = y[test_idx]

#---------------------------------------------------------
#--------------------------PART E-------------------------
#---------------------------------------------------------

print("\n---------Part E: Determinants-------")

# TODO: Compute X_train^T X_train and its determinant
XtX_train = X_train.T @ X_train  # TODO
det_XtX_train = np.linalg.det(XtX_train)   # TODO: use a NumPy linear-algebra function
print(f"det(X_train^T X_train) = {det_XtX_train}")

#---------------------------------------------------------
#--------------------------PART F-------------------------
#---------------------------------------------------------

print("\n---------Part F: Eigenvalues and Eigenvectors-------")
# TODO: Compute the covariance matrix C of your (standardized) training features.
C = np.cov(X_train, rowvar=False)   # TODO: e.g. np.cov(X_train, rowvar=False)
print("Covariance matrix:\n", C)
print("C shape:", C.shape)
print("Symmetric?", np.allclose(C, C.T))


# TODO: Verify C v = lambda v for the leading eigenpair
eigvals, eigvecs = np.linalg.eigh(C)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

v1 = eigvecs[:, 0]
lam1 = eigvals[0]
error = np.max(np.abs(C @ v1 - lam1 * v1))
print("max|C v1 - lambda1 v1| =", error)
print(f"Eigenvalues: {eigvals}")      
print(f"Eigenvectors: \n{eigvecs}")

# TODO: Compute the explained variance ratio for each component and
# inspect which original features load most heavily on the first eigenvector.

# Report how much variance each component explains.
explained_variance_ratio = eigvals / eigvals.sum()
for component, ratio in enumerate(explained_variance_ratio, start=1):
    print(f"PC{component} explains {ratio:.2%} of the variance")

# Inspect how strongly each original feature contributes to PC1.
for feature, weight in zip(feature_names, v1):
    print(f"{feature} weight in PC1: {weight:+.4f}")

#---------------------------------------------------------
#--------------------------PART F-------------------------
#---------------------------------------------------------

# Pair each eigenvector column with its eigenvalue on the diagonal.
V = eigvecs
Lambda = np.diag(eigvals)

# Check that the eigenvectors are orthonormal and reconstruct the covariance.
reconstructed_C = V @ Lambda @ V.T
print("V^T V = I ?", np.allclose(V.T @ V, np.eye(V.shape[1])))
print("Reconstruction matches C ?", np.allclose(C, reconstructed_C))

# Keep the fewest components that explain at least 95% of training variance.
cum_var = np.cumsum(explained_variance_ratio)
k = np.searchsorted(cum_var, 0.95) + 1
print(f"k = {k}, cumulative variance explained = {cum_var[k-1]:.2%}")

# Express each observation in the same principal-component coordinates.
V_k = V[:, :k]
Z_train = X_train @ V_k
Z_test = X_test @ V_k
print("Z_train shape:", Z_train.shape)

# Optional when k >= 2: plot the training observations, colored by chicken price.
# plt.scatter(Z_train[:, 0], Z_train[:, 1], c=y_train, cmap="viridis")
# plt.xlabel("PC1"); plt.ylabel("PC2"); plt.colorbar(label="target y")
# plt.title("Data projected onto the eigenbasis")
# plt.show()
