import sklearn.metrics as met
import tensorflow as tf
from imagenet_c import corrupt
import numpy as np
import cv2

class Robustness():
    def __init__(self, model):
        self.model = model
        self.corruptions = np.array(['gaussian_noise', 'defocus_blur', 'snow', 'contrast'])
        self.baseline = np.array([0.886, 0.82, 0.867, 0.853])
        
    def c_robustness(self, dataset, nb_samples):
        x, y_true = next(iter(dataset.valid_data.batch(dataset.valid_data.cardinality())))
        y_samples, y_pred = self.model.predict(x, nb_samples)
        if dataset.likelihood_model == "Classification":
            accuracy = met.accuracy_score(y_true, tf.argmax(y_pred, axis = 1))
            e_clean = 1 - accuracy
            e_s_c = np.array([np.array([self.error_rate(x, y_true, s, c, nb_samples) for s in range(1, 6)]) for c in self.corruptions])
            ce = np.array([np.sum(e_s_c[i]) / self.baseline[i] for i in range(len(self.corruptions))])
            mce = np.mean(ce)
            relative_ce = np.array([np.sum(e_s_c[i] - e_clean) / (self.baseline[i] - e_clean) for i in range(len(self.corruptions))])
            mrce = np.mean(relative_ce)
            print("Mean Corruption Error", mce)
            print("Mean Relative Error", mrce)
            return mce, mrce
            
    
    def error_rate(self, x, y_true, severity, corruption, nb_samples):
        x_array = x.numpy()
        _, width, length, _ = np.shape(x_array)
        resized = [cv2.resize(image, (224, 224)) for image in x_array]
        corrupted_images_resized = [corrupt(image, severity, corruption) for image in resized]
        corrupted_images = [cv2.resize(image, (width, length)) for image in corrupted_images_resized]

        corrupted_inputs = tf.convert_to_tensor(corrupted_images)
        corrupted_inputs = tf.concat(corrupted_inputs, axis=1)
        _, c_predicted = self.model.predict(corrupted_inputs, nb_samples)
        accuracy = met.accuracy_score(y_true, tf.argmax(c_predicted, axis = 1))
        error = 1 - accuracy
        return error
            