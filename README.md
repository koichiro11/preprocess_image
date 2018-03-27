# pre-process for image recognition

## Purpose

implement some arts to pre-process for image recognition

## Author

- koichiro tamura
- hiromi nakagawa

## setting

### pip
- tensorflow (ver 1.4 recommended)

### conda
- joblib
- scikit-learn
- pillow


## How to use

![flow](https://user-images.githubusercontent.com/12594363/37944837-2fb92500-31b8-11e8-9672-feff189021a1.png)

### 1. set hyperparameter class

First, you choose Hyperparameter class to choose.

For example, when you use CIFAR-10, you set Hyperparameter class in hyperparameter.py as follows:

```
class HyperParameter(HyperParameterCIFAR10):
```

Make sure you define correct hyper-parameter.


### 2. load data

Second, you load data to use and create TFRecord for training.
Before training, execute the `file` as follows.

Make sure you use correct class corresponding to the dataset.

```
$ python load_data.py
```

```python

from data_loader import DataLoaderCIFAR10 as DataLoader

if __name__ == '__main__':

    data_loader = DataLoader()
    data_loader.main()

```load_data.py


when you use specific test data, you use as follows.

```python

from PIL import Image
from data_loader import DataLoader
IMAGE_PATH = 'xxx'

if __name__ == '__main__':

    data_loader = DataLoader()
    with Image.open(IMAGE_PATH) as image:
        image = data_loader.convert_from_pil_into_numpy(image, rgb=True)

```

### 3. preprocess data


Third, you load TFRecord and do data augmentation.

```
$ python preprocess_data.py
```

```python
import tensorflow as tf
from preprocessor import PreProcessorWithAugmentation

if __name__ == '__main__':

    preprocess = PreProcessorWithAugmentation()
    train_path = Path('train*.tfrecord')
    image_size = {
        'height',
        'width',
        'channel',
    }

    # load dataset
    train_dataset = preprocess.load_tfrecords_dataset(train_path, image_size, 10)

    # define iterator
    train_iterator = preprocess.get_iterator(
        train_dataset, batch_size=batch_size, num_epochs=args.num_epochs, buffer_size=100*batch_size, aug_func=train_aug_func, aug_kwargs=aug_kwargs)

    train_batch = train_iterator.get_next()

    # sess
    with tf.Session as sess:
        for i in range(math.ceil(train_size / total_batch_size)):
            train_X_mb, train_Y_mb = sess.run(train_batch)

```preprocess_data.py

