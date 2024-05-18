from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import zipfile
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
from airflow import DAG
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 5, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    'fashion_mnist_preprocessing',
    default_args=default_args,
    description='Download and preprocess Fashion MNIST data using Airflow, Spark, and TensorFlow',
    schedule=None,
)

# Task 1: Download Fashion MNIST Data
def download_fashion_mnist(**kwargs):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    
    logging.info("working")
    train_file_path = '/Users/anikbhowmick/Python/Final_project/airflow_1/Data/train_data.npz'
    test_file_path = '/Users/anikbhowmick/Python/Final_project/airflow_1/Data/test_data.npz'

    # Save the arrays
    np.savez(train_file_path, images=train_images, labels=train_labels)
    np.savez(test_file_path, images=test_images, labels=test_labels)

download_task = PythonOperator(
    task_id='download_fashion_mnist',
    python_callable=download_fashion_mnist,
    provide_context=True,
    dag=dag,
)




# Task 2: Preprocess and Augment Data with Spark and TensorFlow
def preprocess_with_spark(**kwargs):
    spark = SparkSession.builder.appName("FashionMNISTPreprocessing").getOrCreate()

    # Load dataset
    data = np.load('/Users/anikbhowmick/Python/Final_project/airflow_1/Data/train_data.npz')
    x_train, y_train = data['images'], data['labels']

    # Create DataFrame
    train_df = spark.createDataFrame([(x.flatten().tolist(), int(y)) for x, y in zip(x_train, y_train)], ['image', 'label'])

    # Data augmentation function
    def augment_image(image):
        img_array = np.array(image, dtype=np.uint8).reshape(28, 28, 1)
        datagen = ImageDataGenerator(
            rotation_range=60,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        img_aug = next(datagen.flow(img_array[np.newaxis, ...], batch_size=1))[0].astype(np.float32).flatten().tolist()
        return img_aug

    augment_udf = udf(augment_image, ArrayType(FloatType()))

    # Apply augmentation
    augmented_df = train_df.withColumn('augmented_image', augment_udf(train_df['image']))

    # Collect augmented data
    augmented_data = augmented_df.select('augmented_image', 'label').collect()

    # Convert to numpy arrays
    x_augmented = np.array([row['augmented_image'] for row in augmented_data], dtype=np.float32).reshape(-1, 28, 28)
    y_augmented = np.array([row['label'] for row in augmented_data], dtype=np.int32)

    # Save augmented data
    np.savez('/Users/anikbhowmick/Python/Final_project/airflow_1/Augmented_Data/augmented_fashion_mnist.npz', x_train=x_augmented, y_train=y_augmented)

    spark.stop()

preprocess_task = PythonOperator(
    task_id='preprocess_with_spark',
    python_callable=preprocess_with_spark,
    provide_context=True,
    dag=dag,
)

# Task dependencies
download_task >> preprocess_task

