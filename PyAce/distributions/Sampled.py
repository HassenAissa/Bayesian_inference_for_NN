import tensorflow as tf
from distributions.Distribution import Distribution
import os, json
import random
import bisect


class Sampled(Distribution):
    def __init__(self, samples: list[tf.Tensor], frequencies: list[int]):
        super().__init__(int(samples[0].shape[0]))
        if len(samples) == 0:
            raise ValueError("Can't have distribution Sampled with 0 samples")
        if len(samples) != len(frequencies):
            raise ValueError("Number of samples and list frequency do not have the same size")
        if len(samples[0].shape) > 1:
            raise ValueError("Samples must have only one dimension")
        self._n_samples = len(samples)
        # copy list for encapsulation
        self._samples = [s for s in samples]
        self._frequencies = [f for f in frequencies]
        self._acc_frequencies = []
        acc = 0
        for f in self._frequencies:
            acc += f
            if f == 0:
                raise ValueError("Samples frequencies can't sum up to zero")
            self._acc_frequencies.append(acc)

    def sample(self) -> tf.Tensor:
        w = random.randint(1, self._acc_frequencies[self._n_samples - 1])
        index = bisect.bisect_left(self._acc_frequencies, w)
        return self._samples[index]

    def store(self, path: str):
        info = {"size": self._size,
                "n_samples": self._n_samples,
                "frequencies": self._frequencies,
                "dtypes": []}
        for sample in self._samples:
            info["dtypes"].append(sample.dtype.name)
        with open(os.path.join(path, "info.json"), "w") as file:
            file.write(json.dumps(info))
        if not os.path.exists(os.path.join(path, "samples")):
            os.makedirs(os.path.join(path, "samples"))
        sample_path = os.path.join(path, "samples")
        for i, sample in zip(range(self._n_samples), self._samples):
            serialized_sample = tf.io.serialize_tensor(sample)
            tf.io.write_file(os.path.join(sample_path, "sample" + str(i) + ".tf"), serialized_sample)

    @classmethod
    def load(cls, path: str) -> 'Distribution':
        with open(os.path.join(path, "info.json"), "r") as file:
            info = json.load(file)
        sample_dir = os.path.join(path, "samples")
        samples = []
        dtypes = info["dtypes"]
        for i in range(info["n_samples"]):
            serialized_sample = tf.io.read_file(os.path.join(sample_dir, "sample" + str(i)+".tf"))
            samples.append(tf.io.parse_tensor(serialized_sample, getattr(tf, dtypes[i])))
        return Sampled(samples, info["frequencies"])


