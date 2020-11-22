# TensorFlow benchmarks

## Some simple modification
[tf_cnn_benchmarks](https://github.com/scaomath/tf-benchmarks/tree/master/tf_cnn_benchmarks), some simple quick run on CUDA gpu as follows:

1. quick run using 64 images per batch in FP32 
```bash
 python3 ./tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 64
```
2. FP16 gradient with params stored using FP32
```bash
 python3 ./tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --use_fp16 1 --fp16_enable_auto_loss_scale 1 --fp16_loss_scale 1000 --batch_size 64
```
3. Forcing cpu
```bash
 python3 ./tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --device cpu --data_format=NHWC --batch_size 64
```

## Original benchmark by the tf team
This repository contains various TensorFlow benchmarks. Currently, it consists of two projects:


1. [PerfZero](https://github.com/tensorflow/benchmarks/tree/master/perfzero): A benchmark framework for TensorFlow.

2. [tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) (no longer maintained): The TensorFlow CNN benchmarks contain TensorFlow 1 benchmarks for several convolutional neural networks.

If you want to run TensorFlow models and measure their performance, also consider the [TensorFlow Official Models](https://github.com/tensorflow/models/tree/master/official)

