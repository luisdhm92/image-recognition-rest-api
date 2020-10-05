import numpy as np
import tensorflow as tf

from utils.images_fetcher import get_images
from .image_classifier import ImageClassifier


class XceptionImageClassifier(ImageClassifier):

    def classify_images(self, directory: str):

        # todo: fix the input format to be processed by xception

        static_images = get_images(directory)
        model = tf.keras.applications.xception.Xception(weights="imagenet")
        classifications = {
            'results': []
        }

        # error trying to process all images together
        index = 0
        for item in static_images.images:
            images = np.array([item / 255])

            images_resized = tf.image.resize_with_crop_or_pad(images, 224, 224)

            inputs = tf.keras.applications.resnet50.preprocess_input(images_resized * 255)
            Y_proba = model.predict(inputs)

            top_K = tf.keras.applications.resnet50.decode_predictions(Y_proba, top=3)

            for image_index in range(len(images)):
                top_k_items = []
                for class_id, name, y_proba in top_K[image_index]:
                    top_k_items.append(
                        {
                            "class_id": class_id,
                            "name": name,
                            "proba": y_proba * 100
                        }
                    )

                classifications["results"].append(
                    {
                        static_images.filenames[index]: top_k_items
                    }
                )

                index = index + 1

        return classifications