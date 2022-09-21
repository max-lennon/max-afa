import numpy as np
import pickle

def gen_cube(n_features=20, data_points=20000, sigma=0.3, seed=123):
    assert n_features >= 10, 'cube data has >= 10 features'
    np.random.seed(seed)
    clean_points = np.random.binomial(1, 0.5, (data_points, 3))
    labels = np.dot(clean_points, np.array([1,2,4]))
    points = clean_points + np.random.normal(0, sigma, (data_points, 3))
    features = np.random.rand(data_points, n_features)
    for i in range(data_points):
        offset = labels[i]
        for j in range(3):
            features[i, offset + j] = points[i, j]
    return np.array(features, np.float32), np.array(labels, np.float32)

x, y = gen_cube()

indices = list(range(20000))
np.random.shuffle(indices)
train_indices = indices[:10000]
valid_indices = indices[10000:15000]
test_indices = indices[15000:]

train_x, train_y = x[train_indices], y[train_indices]
valid_x, valid_y = x[valid_indices], y[valid_indices]
test_x, test_y = x[test_indices], y[test_indices]

data = {
    'train': (train_x, train_y),
    'valid': (valid_x, valid_y),
    'test': (test_x, test_y)
}

with open('./cube_20_0.3.pkl', 'wb') as f:
    pickle.dump(data, f)