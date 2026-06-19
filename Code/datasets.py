# coding=utf-8
import tensorflow_datasets as tfds
import os


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


def crop_resize(image, resolution):
    """Crop and resize an image to the given resolution."""
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    crop = tf.minimum(h, w)
    image = image[(h - crop) // 2:(h + crop) // 2,
            (w - crop) // 2:(w + crop) // 2]
    image = tf.image.resize(
        image,
        size=(resolution, resolution),
        antialias=True,
        method=tf.image.ResizeMethod.BICUBIC)
    return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
    """Shrink an image to the given resolution."""
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    ratio = resolution / tf.cast(tf.minimum(h, w), tf.float32)
    new_h = tf.cast(tf.cast(h, tf.float32) * ratio, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * ratio, tf.int32)
    return tf.image.resize(image, [new_h, new_w], antialias=True)


def central_crop(image, size):
    """Crop the center of an image to the given size."""
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    top = (h - size) // 2
    left = (w - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)


class LocalCelebABuilder:
    """Custom dataset builder for local CelebA files."""

    def __init__(self):
        self.image_dir = 'your path'
        self.partition_file = 'your path'
        self._split_files = self._get_split_files()

    def _get_split_files(self):
        splits = {'train': [], 'validation': [], 'test': []}
        with open(self.partition_file, 'r') as f:
            for line in f:
                filename, partition = line.strip().split()
                partition = int(partition)
                if partition == 0:
                    splits['train'].append(os.path.join(self.image_dir, filename))
                elif partition == 1:
                    splits['validation'].append(os.path.join(self.image_dir, filename))
                else:
                    splits['test'].append(os.path.join(self.image_dir, filename))
        return splits

    def as_dataset(self, split, shuffle_files=False):
        files = self._split_files[split]
        ds = tf.data.Dataset.from_tensor_slices(files)

        def load_image(file_path):
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=3)
            return {'image': img}

        ds = ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if shuffle_files:
            ds = ds.shuffle(len(files))
        return ds


def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation."""
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    if batch_size % 1 != 0:
        raise ValueError(f'Batch sizes ({batch_size} must be divided by')

    shuffle_buffer_size = 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1

    if config.data.dataset == 'CIFAR10':
        dataset_builder = tfds.builder('cifar10')
        train_split_name = 'train'
        eval_split_name = 'test'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

    elif config.data.dataset == 'CELEBA':
        dataset_builder = LocalCelebABuilder()
        train_split_name = 'train'
        eval_split_name = 'validation'

        def resize_op(img):
            img = tf.ensure_shape(img, [None, None, 3])
            img = tf.image.convert_image_dtype(img, tf.float32)
            # img = central_crop(img, 140)

            img = tf.image.resize(img, [config.data.image_size, config.data.image_size])

            return img


    def preprocess_fn(d):
        """Basic preprocessing function scales data to [0, 1) and randomly flips."""
        img = resize_op(d['image'])
        if config.data.random_flip and not evaluation:
            img = tf.image.random_flip_left_right(img)
        if uniform_dequantization:
            img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

        return dict(image=img, label=d.get('label', None))

    def create_dataset(dataset_builder, split):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.experimental_threading.private_threadpool_size = 48
        dataset_options.experimental_threading.max_intra_op_parallelism = 1

        if isinstance(dataset_builder, LocalCelebABuilder):
            ds = dataset_builder.as_dataset(split=split, shuffle_files=False)
        elif isinstance(dataset_builder, tfds.core.DatasetBuilder):
            read_config = tfds.ReadConfig(options=dataset_options)
            dataset_builder.download_and_prepare()
            ds = dataset_builder.as_dataset(
                split=split, shuffle_files=False, read_config=read_config)
        else:
            ds = dataset_builder.with_options(dataset_options)

        ds = ds.apply(tf.data.experimental.ignore_errors())

        ds = ds.repeat(count=num_epochs)
        # ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(prefetch_size)

    train_ds = create_dataset(dataset_builder, train_split_name)
    eval_ds = create_dataset(dataset_builder, eval_split_name)
    return train_ds, eval_ds, dataset_builder