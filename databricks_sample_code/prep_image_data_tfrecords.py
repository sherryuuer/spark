from pyspark.sql.functions import col, pandas_udf

import os
import uuid
import tensorflow as tf

# load data using Spark
df = spark.read.format("delta").load("/databricks-datasets/flowers/delta")

labels = df.select(col("label")).distinct().collect()
label_to_idx = {label: index for index, (label, ) in enumerate(sorted(labels))}


@pandas_udf("long")
def get_label_idx(labels):
    return labels.map(lambda label: label_to_idx[label])


df = df.withColumn("label_index", get_label_idx(col("label"))) \
       .select(col("content"), col("label_index")) \
       .limit(100)

# Save the data to TFRecord files
name_uuid = str(uuid.uuid4())
path = '/ml/flowersData/df-{}.tfrecord'.format(name_uuid)
df.limit(100).write.format("tfrecords").mode("overwrite").save(path)
display(dbutils.fs.ls(path))

# Load TFRecords using TensorFlow
# Step 1: Create a TFRecordDataset as an input pipeline.
filenames = [("/dbfs" + path + "/" + name)
             for name in os.listdir("/dbfs" + path) if name.startswith("part")]
dataset = tf.data.TFRecordDataset(filenames)


# Step 2: Define a decoder to read, parse, and normalize the data.
def decode_and_normalize(serialized_example):
    """
    Decode and normalize an image and label from the given `serialized_example`.
    It is used as a map function for `dataset.map`
    """
    IMAGE_SIZE = 224

    # 1. define a parser
    feature_dataset = tf.io.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'content': tf.io.FixedLenFeature([], tf.string),
            'label_index': tf.io.FixedLenFeature([], tf.int64),
        })
    # 2. decode the data
    image = tf.io.decode_jpeg(feature_dataset['content'])
    label = tf.cast(feature_dataset['label_index'], tf.int32)
    # 3. resize
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    # 4. normalize the data
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, label


parsed_dataset = dataset.map(decode_and_normalize)


# Use the dataset as an input to train the model.
batch_size = 4
parsed_dataset = parsed_dataset.shuffle(40)
parsed_dataset = parsed_dataset.repeat(2)
parsed_dataset = parsed_dataset.batch(batch_size)
