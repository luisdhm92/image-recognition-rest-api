import abc


class ImageClassifier:

    @abc.abstractmethod
    def classify_images(self, directory: str) -> dict:
        pass
