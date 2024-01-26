import tensorflow as tf
from src.distributions.Distribution import Distribution
import os, json

class SampledDistribution(Distribution):
    def __init__(self, param_vecs: list[tf.Tensor], probs, sample_index: int):
        super().__init__(int(param_vecs[0].shape[0]))
        self.index = sample_index
        self.n_samples = len(param_vecs)
        self.param_vecs = param_vecs
        self.probs = probs
        self.probs_acc = []
        p_acc = 0
        for p in probs:
            p_acc += p
            self.probs_acc.append(p_acc)
        # make sure probabilty sums to 1
        if p_acc != 1:
            raise ValueError("Samples probabilties don't sum to 100%")

    def sample(self) -> tf.Tensor:
        p_acc = tf.random.uniform(shape=[])
        for i in range(self.n_samples):
            if self.probs_acc[i] > p_acc:
                return self.param_vecs[i]

    def serialize(self) -> str:
        info = {"sample_index": self.index, 
                "parameter dimension": self._size,
                "capacity": self.n_samples,
                "probabilities": self.probs}
        f = open("src/distributions/dtbn_infos.json", "a")
        content = json.dumps(info)
        f.write(content+'\n')
        f.close()
        for i in self.n_samples:
            f = open("src/distributions/samples/sd"+str(self.index)+"_sample"+str(i)+".npy", "wb") 
            # example "sd5_sample2.npy"
            f.write(self.param_vecs[i]) # ? how to write tensor to .npy files ?
            f.close()
        return content

    @classmethod
    def deserialize(cls, data: str) -> 'Distribution':
        # need a sample index input
        sample_index = 5
        f = open("src/distributions/dtbn_infos.json", "r")
        res = ""
        start = False
        while True:
            line = f.readline()
            if "sample_index" in line and str(sample_index) in line:
                if not start:
                    start = True
                else:
                    break
            if start:
                res += line

        info = json.loads(res)
        return SampledDistribution()    # To be completed
        
