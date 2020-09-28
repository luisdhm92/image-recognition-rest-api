from flask import jsonify, Blueprint

from models.cnn.image_classifier_factory import ImageClassifierFactory

REQUEST_API = Blueprint('cnn_api', __name__)


def get_blueprint():
    """Return the blueprint for the main app module"""
    return REQUEST_API


@REQUEST_API.route('/classify-images', methods=['GET'])
def classify_images():
    """Classify the images in the static directory
    @return: 200: an array with the image classifications as
    flask/response object with application/json mimetype.
    """
    images_classifier = ImageClassifierFactory.get_image_classifier(model='restnet50')
    classifications = images_classifier.classify_images('./static/img')
    return jsonify(classifications)
