from models.cnn.image_classifier import ImageClassifier
from models.cnn.restnet50 import RestNetImageClassifier
from models.cnn.xception import XceptionImageClassifier


class ImageClassifierFactory:

    @staticmethod
    def get_image_classifier(model='') -> ImageClassifier:

        if model is 'xception':
            classifier = XceptionImageClassifier()
        else:
            # default model restnet50
            classifier = RestNetImageClassifier()

        return classifier
