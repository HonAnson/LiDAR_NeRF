import numpy as np
from visualize import getMask, quat2RotationMatrix, getPsudoImgVec
from utility import quickVizNumpy, quickVizTwoNumpy
import cv2
import numpy as np
import pywt
import pandas as pd

# Example feature data
np.random.seed(42)
X = np.random.uniform(-1, 1, (100, 5))  # 100 samples, 5 features

# Wavelet Transform Function
def wavelet_transform_features(X, wavelet='db1', level=3):
    n_samples, n_features = X.shape
    transformed_features = []

    for i in range(n_features):
        coeffs = pywt.wavedec(X[:, i], wavelet, level=level)
        transformed_feature = np.concatenate(coeffs)
        transformed_features.append(transformed_feature)

    transformed_features = np.array(transformed_features).T
    return transformed_features

# Apply wavelet transform to the features
X_wavelet_transformed = wavelet_transform_features(X, wavelet='db1', level=3)

# Convert to DataFrame for better visualization
df_wavelet_transformed = pd.DataFrame(X_wavelet_transformed)
print(df_wavelet_transformed.head())

# Output shape
print("Original shape:", X.shape)
print("Transformed shape:", X_wavelet_transformed.shape)




# if __name__ == "__main__":
#     camear_direction = np.array([1,0,0])
#     pixel_to_meter = 500
#     img_width = 500
#     img_height = 500
#     focal_length = 1.43580442
#     vec = getPsudoImgVec(camear_direction, pixel_to_meter, img_width, img_height, focal_length)
    


