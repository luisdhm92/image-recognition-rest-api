import os
from sklearn.externals._pilutil import imread
from sklearn.utils import Bunch


def get_images(url='../static/img', file_extension='.jpg'):
    with open(f"{url}/README.txt") as f:
        descr = f.read()

    filenames = [
        f"{url}/{filename}"
        for filename in sorted(os.listdir(url)) if filename.endswith(file_extension)
    ]
    images = [imread(filename) for filename in filenames]
    return Bunch(
        images=images,
        filenames=filenames,
        DESC=descr
    )
