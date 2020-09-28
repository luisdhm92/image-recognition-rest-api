from models.cnn.image_classifier import ImageClassifier
from models.cnn.restnet50 import RestNetImageClassifier


class ImageClassifierFactory:

    @staticmethod
    def get_image_classifier(model='') -> ImageClassifier:

        if model is 'lenet-5':
            classifier = RestNetImageClassifier()
        elif model is 'xception':
            classifier = RestNetImageClassifier()
        else:
            # default model restnet50
            classifier = RestNetImageClassifier()

        return classifier
