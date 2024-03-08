import sklearn.metrics as met
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage as sk
import os

# //////// Corruptions ////////
def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def gaussian_noise_regression(x_col, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    
    return np.array(x_col) + np.random.normal(loc=0, scale=c, size=x_col.shape)

def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255

def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = sk.filters.gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255

def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255

def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.

    if len(x.shape) < 3 or x.shape[2] < 3:
        x = np.clip(x + c, 0, 1)
    else:
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255

def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.

    gray_scale = False
    if len(x.shape) < 3 or x.shape[2] < 3:
        x = np.array([x, x, x]).transpose((1, 2, 0))
        gray_scale = True
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    if gray_scale:
        x = x[:, :, 0]

    return np.clip(x, 0, 1) * 255

def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x_shape = np.array(x).shape

    x = x.resize((int(x_shape[1] * c), int(x_shape[0] * c)), Image.BOX)

    x = x.resize((x_shape[1], x_shape[0]), Image.NEAREST)

    return x

class Robustness():
    """
        a class representing the robustness analysis of a model
        stores the errors for different corruptions and severities as it computes them to save computations
    """
    def __init__(self, model, dataset):
        self.model = model
        self.regression = dataset.likelihood_model == "Regression"
        self.corruptions = (gaussian_noise, shot_noise, impulse_noise, speckle_noise,
                            gaussian_blur, contrast, brightness, saturate, pixelate)
        self.corruption_dict = {corr_func.__name__: corr_func for corr_func in self.corruptions}
        self.error_dict = {c: None for c in self.corruption_dict.keys()}
        self.n = len(self.corruptions)
        self.severities = np.arange(1, 6)
        self.dataset = dataset
        self.x, self.y_true = next(iter(self.dataset.valid_data.batch(self.dataset.valid_data.cardinality())))
        #self.x = tf.cast(self.x, tf.float32)



    def adversarial_robustness(self, epsilon=0.1, nb_samples=100, save_path=None):
        """
        computes the adversarial robustness of the model using a FGSM gradient attack

        Args:
            epsilon (float, optional): Severity of the attack. Defaults to 0.1.
            nb_samples (int, optional): Defaults to 100.
            save_path (_type_, optional): Defaults to None.
        """
        x_grad = 0
        for i in range(nb_samples):
            sample_model = self.model.sample_model()
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.x)
                #print(self.x.shape)
                prediction = sample_model(self.x, training=True)
                prediction = tf.where(tf.math.is_nan(prediction), tf.zeros_like(prediction), prediction)
                loss = self.dataset.loss()(self.y_true, prediction)
                #print(met.accuracy_score(self.y_true, tf.argmax(prediction, axis = 1)) * 100)
            x_temp_grad = tape.gradient(loss, self.x)
            x_grad += x_temp_grad
        x_perturbated = self.x + epsilon * np.sign(x_grad)
        #print((epsilon * np.sign(x_grad))[0])
        sample_preds, predicted = self.model.predict(x_perturbated, nb_samples)
        if self.regression:
            robustness = met.root_mean_squared_error(self.y_true, predicted)
            stat = "Adversarial Robustness: " + str(robustness)
        else:
            robustness = met.accuracy_score(self.y_true, tf.argmax(predicted, axis = 1)) * 100
            stat = "Adversarial Robustness: " + str(robustness) + "%"
        self._save_data(save_path, "adversarial_robustness", robustness) if save_path else print(stat)


    def mean_corruption_error(self, relative=False, nb_samples=100, save_path=None):
        """
        Computes de mean corruption error and realtive corruption error across all corruptions and severities
        Prints value or saves it if save_path param given

        Args:
            relative (bool, optional): computes relative error if set to True. Defaults to False.
            nb_samples (int, optional): Defaults to 100.
            save_path (_type_, optional): pass in path to save value on file. Defaults to None.
        """
        if self.regression:
            ce = np.array([self.corruption_error(helper=True, relative=relative)])
        else:
            ce = np.array([self.corruption_error(c, helper=True, relative=relative) for c in self.corruption_dict.keys()])
        mean = np.mean(ce)
        if save_path:
            name = "mean_relative_error" if relative else "mean_corruption_error"
            self._save_data(save_path, name, mean)
        else:
            name = "Mean Relative Error: " if relative else "Mean Corruption Error: "
            stat = name + str(mean) if self.regression else name + str(mean) + " %"
            print(stat)
    
    def corruption_error(self, corruption=None, relative=False, nb_samples=100, save_path=None, helper=False):
        """
        Computes the corruption error across all severities for a specific corruption

        Args:
            corruption (String): corruption 
            relative (bool, optional): if set to True, computes relative corruption error. Defaults to False.
            nb_samples (int, optional): _description_. Defaults to 100.
            save_path (_type_, optional): _description_. Defaults to None.
            helper (bool, optional): set to True only when used internally as a helper method. Defaults to False.
        """
        if self.regression:
            error = [np.array([self._error_rate(s, nb_samples) for s in self.severities])]
        else:    
            if self.error_dict[corruption] :
                error = self.error_dict[corruption]
            else:
                error = [np.array([self._error_rate(s, nb_samples, corruption)*100 for s in self.severities])]
                self.error_dict[corruption] = error
        if relative:
            _, y_pred = self.model.predict(self.x, nb_samples)
            clean_error = met.root_mean_squared_error(self.y_true, y_pred) if self.regression else 1 - met.accuracy_score(self.y_true, tf.argmax(y_pred, axis = 1))
            ce = np.sum([x-clean_error for x in error]) / len(self.severities)
        else:
            ce = np.sum(error) / len(self.severities)
        if helper:
            return ce
        if save_path:
            name_file = "relative_error_" if relative else "corruption_error_"
            self._save_data(save_path, name_file + corruption, ce)
        else:
            print_stat = "Relative Error for {}: {}%" if relative else "Corruption Error for {}: {}%"
            print(print_stat.format(corruption, ce))
       
    def robustness_by_corruption(self, nb_samples=100, save_path=None):
        """
        Plots corruption error as a bar graph giving the robustness score for each corruption.
        """
        ce = np.array([self.corruption_error(c, nb_samples=nb_samples, helper=True) for c in self.corruption_dict.keys()])
        plt.bar(self.corruptions_dict.keys(), ce)
        plt.xlabel("Corruption")
        label = "Corruption Error" if self.regression else "Corruption Error (%)"
        plt.ylabel(label)
        plt.title("Corruption Error by Corruption")
        self._save_figure(save_path, "robustness_by_corruption") if save_path else plt.show()
        
    def corruption_robustness_by_severity(self, corruption=None, nb_samples=100, save_path=None):
        """
        Plots corruption error for given corruption as a function of the severity
        """
        if self.regression:
            error = [self._error_rate(s, nb_samples) for s in self.severities]
        else:
            if self.error_dict[corruption] :
                error = self.error_dict[corruption]
            else:
                error = [self._error_rate(s, nb_samples, corruption) for s in self.severities]
                self.error_dict[corruption] = error
        plt.bar(self.severities, error)
        plt.xlabel("Severity")
        label = "Error Rate" if self.regression else "Error Rate (%)"
        plt.ylabel(label)
        plt.title("Error Rate by Severity for {}".format("noise" if self.regression else corruption))
        self._save_figure(save_path, "corruption_robustness_by_severity") if save_path else plt.show()
    
    def _error_rate(self, severity, nb_samples, corruption=None):
        input = self.x.numpy()
        if self.regression:
            corrupted_inputs = [self._corrupt_regression(x, severity) for x in input]
        else:
            colour_scale = np.shape(input)[-1]
            #_, width, length, _ = np.shape(x_array)
            #resized = [cv2.resize(image, (224, 224)) for image in x_array]
            corrupted_inputs = [self._corrupt(image, severity, corruption) for image in input]
            #corrupted_images = [cv2.resize(image, (width, length)) for image in corrupted_images_resized]
            if colour_scale == 1:
                corrupted_inputs = np.stack([np.mean(image, axis=-1, keepdims=True) for image in corrupted_inputs])
        corrupted_inputs = tf.convert_to_tensor(corrupted_inputs)
        corrupted_inputs = tf.concat(corrupted_inputs, axis=1)
        _, c_predicted = self.model.predict(corrupted_inputs, nb_samples)
        if self.regression:
            error = met.root_mean_squared_error(self.y_true, c_predicted)
        else:
            accuracy = met.accuracy_score(self.y_true, tf.argmax(c_predicted, axis = 1))
            error = 1 - accuracy
        #print("Error rate for", corruption, ", with severity", severity, ":", error)
        return error

    # Image width and height must be at least 32 pixels
    def _corrupt(self, image, severity, corruption):
        if image.ndim == 2:
            image = np.stack((image,)*3, axis=-1) 
        
        height, width, channels = image.shape
        
        if channels == 1:
            image = np.stack((np.squeeze(image),)*3, axis=-1)

        image_corrupted = self.corruption_dict[corruption](Image.fromarray(image), severity)
        
        return np.uint8(image_corrupted)
    
    def _corrupt_regression(self, x, severity):
        return np.array([gaussian_noise_regression(col, severity) for col in x])
    
    def _save_figure(self, path, name):
        directory = path + "/report/robustness"
        figures = directory + "/figures"
        os.makedirs(directory, exist_ok=True)
        os.makedirs(figures, exist_ok=True)
        plt.savefig(figures + "/" + name + ".png")
        
    def _save_data(self, path, name, content):
        directory = path + "/report/robustness"
        os.makedirs(directory, exist_ok=True)
        file_path = directory + "/" + name + ".txt"
        f = open(file_path, "w")
        f.write(str(content))
        f.close()