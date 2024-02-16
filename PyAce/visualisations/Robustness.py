import sklearn.metrics as met
import tensorflow as tf
# from imagenet_c import corrupt
import matplotlib.pyplot as plt
import numpy as np
# import cv2

class Robustness():
    """
        a class representing the robustness analysis of a model
    """
    def __init__(self, model):
        self.model = model
        self.corruptions = np.array(['gaussian_noise', 'defocus_blur', 'snow', 'contrast'])
        self.n = len(self.corruptions)
        self.severities = np.arange(1, 6)
        self.baseline = np.array([0.886, 0.82, 0.867, 0.853])
        self.baseline_clean = 0.164
    
    def c_robustness(self, dataset, nb_samples):
        """
        outputs visualisaitons for the corruption robustness analysis. Applies some corruptions to datasets, and outputs accuracy and error rates on predictions.

        Args:
            dataset (Dataset): dataset to perform analysis on
            nb_samples (int): number of samples
        """
        x, y_true = next(iter(dataset.valid_data.batch(dataset.valid_data.cardinality())))
        _, y_pred = self.model.predict(x, nb_samples)
        if dataset.likelihood_model == "Classification":
            accuracy = met.accuracy_score(y_true, tf.argmax(y_pred, axis = 1))
            e_clean = 1 - accuracy
            e_s_c = np.array([np.array([self.error_rate(x, y_true, s, c, nb_samples) for s in self.severities]) for c in self.corruptions])
            #e_s_c = np.array([np.array([0.8616, 0.876, 0.8796, 0.887, 0.9002]), 
            #                  np.array([0.8418, 0.8418, 0.8422000000000001, 0.8398, 0.835]),
            #                  np.array([0.852, 0.8588, 0.8626, 0.8704000000000001, 0.8646]),
            #                  np.array([0.849, 0.8662, 0.8882, 0.8986, 0.8986])])
            ce = np.array([np.sum(e_s_c[i]) / (self.baseline[i] * 5) for i in range(self.n)])
            mce = np.mean(ce)
            relative_ce = np.array([np.sum(e_s_c[i] - e_clean) / (self.baseline[i] - self.baseline_clean) * 5 for i in range(self.n)])
            mrce = np.mean(relative_ce)
            print("Mean Corruption Error", mce*100, "%")
            print("Mean Relative Error", mrce*100, "%")
            self.plot_ce_by_corruption(ce)
            for c in self.corruptions:
                self.plot_ce_by_severity(c, e_s_c)
        
    def plot_ce_by_corruption(self, errors):
        plt.bar(self.corruptions, errors*100)
        plt.xlabel("Corruption")
        plt.ylabel("Corruption Error (%)")
        plt.title("Mean Corruption Error by Corruption")
        plt.show()
        
    def plot_ce_by_severity(self, corruption, errors):
        index = np.where(self.corruptions == corruption)
        plt.plot(self.severities, np.reshape(errors[index], 5)*100)
        plt.xlabel("Severity")
        plt.ylabel("Error Rate (%)")
        plt.title("Error Rate by Severity for {}".format(corruption))
        plt.show()
    
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
        #print("Error rate for", corruption, ", with severity", severity, ":", error)
        return error
            