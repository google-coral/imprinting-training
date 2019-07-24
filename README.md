# Mobilenet V1 with L2-Normalized embedding and L2-Normalized weights of fully connected layer for [weights imprinting](https://arxiv.org/pdf/1712.07136.pdf)

## Table of contents

<a href='#Training'>Preparing dependencies</a><br>
<a href='#Training'>Training the model</a><br>
<a href='#Eval'>Evaluating performance</a><br>
<a href='#Export'>Exporting Inference Graph</a><br>

## Preparing dependencies

```shell
$ git submodule init && git submodule update
$ PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/models/research/slim
```

## Training the model

<a id='Training'></a>

The following example demonstrates how to train with L2-Normalized embedding and
L2-Normalized weights of fully connected layer using the default parameters on
the ImageNet dataset.

```shell
$ DATASET_DIR=/tmp/imagenet # Example
$ CHECKPOINT_DIR=/tmp/train_logs # Example
$ FINETUNE_CHECKPOINT_PATH=/tmp/my_checkpoints/mobilenet_v1_quant.ckpt # Example

$ python3 classification/mobilenet_v1_l2norm_train.py \
    --quantize=True \
    --fine_tune_checkpoint=$FINETUNE_CHECKPOINT_PATH \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --dataset_dir=$DATASET_DIR \
    --freeze_base_model=True
```

## Evaluating performance of a model

<a id='Eval'></a> Below we give an example of evaluating it on the imagenet
dataset.

```shell
$ CHECKPOINT_FILE=$CHECKPOINT_DIR/mobilenet_v1_l2norm.ckpt  # Example
$ python3 classification/mobilenet_v1_l2norm_eval.py \
    --quantize=True \
    --checkpoint_dir=$CHECKPOINT_FILE \
    --dataset_dir=$DATASET_DIR
```

## Exporting the Inference Graph

<a id='Export'></a>

Saves out a GraphDef containing the architecture of the model.

To use it, run:

```shell
$ python3 classification/export_inference_graph_l2norm.py \
  --quantize=True \
  --output_file=/tmp/mobilenet_v1_l2norm_inf_graph.pb
```

### Freezing the exported Graph

If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:

```shell
$ cd tensorflow
$ bazel build tensorflow/python/tools:freeze_graph
$ bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=/tmp/mobilenet_v1_l2norm_inf_graph.pb \
  --input_checkpoint=/tmp/checkpoints/mobilenet_v1_l2norm.ckpt \
  --input_binary=true --output_graph=/tmp/frozen_mobilenet_v1_l2norm.pb \
  --output_node_names=MobilenetV1/Predictions/Reshape_1
```

### Transforming the frozen graph

For the model with an L2Norm operator in the convolutional kernels, the frozen
graph cannot be transformed to tflite model successfully. Therefore, we need to
transform the frozen graph to get rid of this operator.(It is feasible because
the weights are constant after training ends.)

```shell
$ bazel build tensorflow/tools/graph_transforms:transform_graph

$ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
  --in_graph=/tmp/frozen_mobilenet_v1_l2norm.pb  \
  --out_graph=/tmp/frozen_mobilenet_v1_l2norm_optimized.pb  \
  --inputs=input --outputs=MobilenetV1/Predictions/Reshape_1 \
  --transforms='strip_unused_nodes
      fold_constants'
```
