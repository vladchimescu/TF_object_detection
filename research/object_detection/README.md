# TensorFlow object detection API for microscopy images
In this repository the object detection API (`tensorflow/models/research/object_detection`) is adapted for using on biological data (e.g. microscopy images).


If you use the TensorFlow Object
Detection API for a research publication, cite the following publication:

```
"Speed/accuracy trade-offs for modern convolutional object detectors."
Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z,
Song Y, Guadarrama S, Murphy K, CVPR 2017
```

## Instructions

1. Prepare your data:
+ Label your images in `input_images` using [tzutalin/labelimg](https://github.com/tzutalin/labelImg)
+ Split the labelled data in `input_images` into train / test folds so that you have
```
input_images/train
input_images/test
```
+ Run `xml_to_csv.py` script: 
```
python xml_to_csv.py input_images
```
+ Generate TensorFlow record files:
```
dir=input_images
python generate_tfrecord.py --csv_input=$dir/train_labels.csv --image_dir=$dir/train --output_path=$dir/train.record
# same for the test set
python generate_tfrecord.py --csv_input=$dir/test_labels.csv  --image_dir=$dir/test --output_path=$dir/test.record
```

2. Set up the object detection API for training:
+ Create a directory `trainmodel`
```
mkdir trainmodel
```
+ Create `labelmap.pbtxt` text file in `trainmodel/` specifying object classes, e.g.
```
item {
  id: 1
  name: 'apoptotic cell'
}

item {
  id: 2
  name: 'viable cell'
}
```
+ Download a pre-trained model from the [TensorFlow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md), e.g.
```
# download
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

# extract
tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

+ Copy the configuration file `pipeline.config` into the `trainmodel` directory:
```
cp faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config trainmodel
```

3. *IMPORTANT*: Modify the configuration file (`trainmodel/pipeline.config`
+ Specify the absolute path to the checkpoint of the pre-trained model:
```
fine_tune_checkpoint: "$ABSOLUTE_PATH/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
```
+ Specify the location of the label map (`trainmodel`) and path to the train data (`PATH_TO_DATA`):
```
train_input_reader {
  label_map_path: "ABS_PATH/trainmodel/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "PATH_TO_DATA/train.record"
  }
}
```

+ Similarly, for the test data:
```
eval_input_reader {
  label_map_path: "ABS_PATH/trainmodel/label_map.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "PATH_TO_DATA/test.record"
  }
```

4. Train the model:
```
python legacy/train.py --logtostderr --train_dir=trainmodel --pipeline_config_path=trainmodel/pipeline.config 
```



