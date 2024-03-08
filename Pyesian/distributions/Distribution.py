from abc import ABC, abstractmethod
import tensorflow as tf



class Distribution(ABC):
    def __init__(self, size: int):
        self._size = size

    @property
    def size(self) -> int:
        """
        returns the site of the distribution

        Returns:
            int: the size of the distribution
        """
        return self._size

    @abstractmethod
    def sample(self) -> tf.Tensor:
        """
        samples from this distribution

        Returns:
            tf.Tensor: the sample
        """
        pass

    @abstractmethod
    def store(self, path: str):
        """stores the distribution

        Args:
            path (str): path to the file to store the distribuition
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'Distribution':
        """loads a distribution

        Args:
            path (str): path of the file from which to load the distribution

        Returns:
            Distribution: The loaded distribution
        """
        pass





