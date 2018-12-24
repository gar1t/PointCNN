# PointCNN and Guild AI

This project supports the following functionality with Guild AI by way
of [guild.yml](guild.yml):

- Track each training run as a separate experiment
- Compare run results
- Diff code changes and hyperparameters across runs
- View runs in TensorBoard
- Train remotely on GPU accelerated systems

## Quick start

Below is a list of commands to perform common project tasks. Refer to
[Overview](#overview) and [Get started](#get-started) below to setup
your environment before running these commands.

| Task           | Dataset     | Commands |
| -------------- | ----------- | -------- |
| Classification | Modelnet    | `$ guild run train-cls dataset=modelnet` |
| Classification | TU Berlin   | `$ guild run prepare-tu-berlin`<br> `$ guild run train-cls dataset=tu_berlin` |
| Classification | Quick Draw  | `$ guild run prepare-quick-draw`<br> `$ guild run train-cls dataset=quick_draw` |
| Classification | MNIST       | `$ guild run prepare-mnist`<br> `$ guild run train-cls dataset=mnist` |
| Classification | CIFAR-10    | `$ guild run prepare-cifar10`<br> `$ guild run train-cls dataset=cifar10` |
| Classification | ScanNet     | *Not supported in guild.yml - refer to [steps in README](README.md#scannet)* |
| Segmentation   | ShapeNet    | `$ guild run prepare-shapenet`<br> `$ guild run train-seg-shapenet`<br> `$ guild run test-seg-shapenet`<br> `$ guild run evaluate-seg-shapenet` |
| Segmentation   | S3DIS       | [Download S3DIS data](http://buildingparser.stanford.edu/dataset.html#Download) before running these commands.<br>`$ guild run prepare-s3dis-labels data=<path to Stanford3dDataset_v1.2_Aligned_Version>`<br> `$ guild run prepare-s3dis-data`<br> `$ guild run prepare-s3dis-filelists`<br> `$ guild run train-seg-s3dis`<br> `$ guild run test-seg-s3dis`<br> `$ guild run merge-s3dis-predictions`<br> `$ guild run evaluate-seg-s3dis` |
| Segmentation   | ScanNet     | *Not supported in guild.yml - refer to [steps in README](README.md#scannet-1)* |
| Segmentation   | Semantic 3D | `$ guild run download-semantic3d`<br> `$ guild run prepare-semantic3d-data`<br> `$ guild run ` |

You can set hyperparameters for any runs using `guild run OPERATION
FLAG=VAL...` where `FLAG` is the name of the hyperparameter. To get
help for the project, including supported flags, run:

    $ guild help

To get help for a specific operation, use `guild run OPERATION
--help-op`. For example, to list available hyperparameters for
`train-cls` (classification task), run:

    $ guild run train-cls --help-op

To use a customized setting module for classification, create a module
in `pointcnn_cls` containing the custom values (we recommend copying
one of the standard setting modules) and specify the module name for
the `setting` flag. For example, if you create
`pointcnn_cls/my_custom_setting.py`, run:

    $ guild run train-cls setting=my_custom_setting

## Overview

[Guild AI](https://guild.ai) is an open source command line tool that
automates project tasks. Guild AI works by reading configuration in
[guild.yml](guild.yml) - it does not require changes to project source
code. Guild AI is similar to tools like Maven or Grunt but with
features supporting machine learning workflow.

Below is a summary of Guild AI commands that can be used with this
project.

**`guild help`** <br> Show project help including models, operation,
and supported flags.

**`guild run [MODEL]:OPERATION [FLAG=VAL]...`** <br> Runs a model
operation. Runs are tracked as unique file system artifacts that can
be managed, inspected, and compared with other runs. Flags may be
specified to change operation behavior.

**`guild runs`** <br> List runs, including run ID, model and
operation, start time, status, and label.

**`guild runs rm RUN`** <br> Delete a run where `RUN` is a run ID or
listing index. You can delete multiple runs matching various criteria.

**`guild compare`** <br> Compare run results including loss and
validation accuracy.

**`guild tensorboard`** <br> View project runs in TensorBoard. You can
view all runs or runs matching various criteria.

**`guild diff RUN1 RUN2`** <br> Diff two runs. You can diff flags,
output, dependencies, and files using a variety of diff tools.

**`guild view`** <br> Open a web based run visualizer to compare and
inspect runs.

For a complete list of commands, run:

```
$ guild --help
```

For help with a specific command, run:

```
$ guild COMMAND --help
```

## Get started

The `guild` program is part of [Guild
AI](https://github.com/guildai/guildai) and can be installed using
pip.

Follow the steps below to install Guild AI and initialize a project
environment.

### Install Guild AI

To install Guild AI, use `pip`:

```
$ pip install guildai --upgrade
```

For additional information, see [Install Guild
AI](https://guild.ai/install/).

### Clone PointCNN repository

```
$ git clone https://github.com/yangyanli/PointCNN.git
```

### Initialize environment

Change to the project directory:

```
$ cd PointCNN
```

Initialize an environment:

```
$ guild init
```

The `init` command creates a virtual environment in `env` and installs
Guild AI and the Python packages listed in
[`requirements.txt`](requirements.txt). Environments are used to
isolate project work from other areas of the system.

Activate the environment:

```
$ source guild-env
```

Check the environment:

```
$ guild check
```

If you get errors, run `guild check --verbose` to get more information
and, if you can't resolve the issue, [open an
issue](https://github.com/guildai/guildai/issues) to get help.

## Train PointCNN for classification on ModelNet 40

To train PointCNN on ModelNet 40, run:

```
$ guild run train-cls dataset=modelnet
```

The default setting for modelnet is `modelnet_x3_l4`.

To start training, press `Enter`.

You can alternatively use a different value for `setting`. To view
available flags and supported values, run:

```
$ guild run train-cls --help-op
```

You may used any of the setting choices that start with `modelnet_`
with the `modelnet` dataset.

## View training progress in TensorBoard

To view training progress in TensorBoard, open a separate command
console.

In the new command console, change to the project directory:

```
$ cd PointCNN
```

Activate the environment:

```
$ source guild-env
```

List project runs:

```
$ guild runs
```

Guild shows the running `train-cls` operation (run ID and dates will
differ):

```
[1:e48c5380]  ../pointcnn:train-cls  2018-11-19 16:17:25  running  modelnet
```

Open TensorBoard:

```
$ guild tensorboard
```

If you run `guild tensorboard` on your workstation, Guild starts
TensorBoard on an available port and opens it in your browser. If you
run the command on a remote server, you have to open TensorBoard in
your browser manually. Use the link displayed in the console.

If you need to run TensorBoard on a specific port, use the `--port`
option:

```
$ guild tensorboard --port 8080
```

Guild automatically synchronizes TensorBoard with the current list of
run. You can leave TensorBoard running during your work.

## Train PointCNN for classification on MNIST

To train a classifier on MNIST, first prepare the MNIST dataset:

```
$ guild run prepare-mnist
```

Press `Enter` to start the operation.

The operation downloads the MNIST data and processes it using
[prepare_mnist_data.py](data_conversions/prepare_mnist_data.py).

View the files generated by the run using `guild ls`:

```
$ guild ls
```

Note that `prepare-mnist` generates `*.txt` and `*.h5` files. These
are used as data inputs to the `train-cls` operation whenever the `mnist`
dataset is specified.

Train PointCNN on MNIST by specifying `mnist` as the dataset:

```
$ guild run train-cls dataset=mnist epochs=10
```

Press `Enter` to start the operation.

The operation uses the dataset files generated by
`prepare-mnist`. This linkage is defined in [guild.yml](guild.yml) as
a resource dependency.

If you have TensorBoard running from the step above, TensorBoard
automatically displays the training progress for the new operation.

### Stop early

The operation is configured to train for 10 epochs (see `epochs=10`
above), however, you can stop early by tying `Ctrl-C` in the operation
console.

Alternative, from a separate console, activate the environment using
`source guild-env` (see above) and run:

```
$ guild stop
```

Guild prompts you to stop running operations. Type `y` and press
`Enter` to stop them.

Training runs may be restarted later, resuming at the most recently
saved checkpoint.

### Restart an operation

You can restart an operation using the `--restart` option:

```
$ guild run --restart <run ID>
```

Replace `<run ID>` with the ID of the run you want to restart.

Training will resume using the latest checkpoint and will continue for
the specified number of epochs (or the default from setting).

You may want to restart a training run for various reasons:

- The run failed with an error that does not effect saved checkpoints
- You want to train over more epochs
- You want to train using different settings (e.g. lower learning
  rate, etc.)

## Compare model performance

You may compare model performance using TensorFlow (see steps above
for starting TensorFlow with Guild) or using the Guild AI `compare`
command.

To compare model loss and validation accuracy, run:

```
$ guild compare
```

Use the arrow keys to navigate within the Compare program.

Press `q` to exit the Compare program.

## Classification on other datasets

### Classification on CIFAR-10

```
$ guild run prepare-cifar10
$ guild run train-cls dataset=cifar10
```

### Classification on Quick Draw!

```
$ guild run prepare-quick-draw
$ guild run train-cls dataset=quick_draw
```

### Classification on TU Berlin

```
$ guild run prepare-tu-berlin
$ guild run train-cls dataset=tu_berlin
```

### Classification on ScanNet

ScanNet is not currently supported in [guild.yml](guild.yml).

## PointCNN segmentation

### Segmentation on ShapeNet

```
$ guild run prepare-shapenet
$ guild run train-seg-shapenet
$ guild run test-seg-shapenet
$ guild run evaluate-seg-shapenet
```

### Segmentation on S3DIS

You must first download the S3DIS dataset, which requires
authorization via a short form.

[Download instructions for S3DIS](https://github.com/alexsax/2D-3D-Semantics#download)

When you have access to the S3DIS dataset files, download
Stanford3dDataset_v1.2_Aligned_Version.zip and unzip it:

```
$ unzip Stanford3dDataset_v1.2_Aligned_Version.zip
```

Note the location of the `Stanford3dDataset_v1.2_Aligned_Version`
directory and set a variable named `DATA_DIR`:

```
$ DATA_DIR=<full path to Stanford3dDataset_v1.2_Aligned_Version>
```

The S3DIS dataset requires three operations before training:

```
$ guild run prepare-s3dis-labels data=$DATA_DIR
$ guild run prepare-s3dis-data
$ guild run prepare-s3dis-filelists
```

Once the dataset is prepared, train PointCNN for segmentation on S3DIS
by running:

```
$ guild run train-seg-s3dis
```

By default area 1 is trained. You can train on other areas by running:

```
$ guild run train-seg-s3dis area=N
```

where `N` is the area you want to train on (supported values 1-6).

You can stop a train operation at any time by pressing `Ctrl-C` in the
operation console, or by running `guild stop` in another console.

Note that if you run commands in a second console, ensure that the
project environment is activated in that console by running `source
guild-env` first.

If you want to restart a training operation from the latest
checkpoint, run:

```
$ guild run --restart RUN_ID
```

`RUN_ID` must be the run ID of the training operation you want to
restart. To get a list of IDs, use `guild runs`.

Once the model is trained, you can test one of its checkpoints using
`test-seg-s3dis`. To get a list of checkpoints for the latest training
operation, run:

```
$ guild ls --operation train-seg-s3dis --path ckpts
```

Note the step of the checkpoint you want to test (e.g. from
`ckpts/iter-STEP.*`) and run:

```
$ guild run test-seg-s3dis step=STEP
```

To evaluate the model performance, you need to run two operations:

```
$ guild run merge-s3dis-predictions
$ guild run evaluate-seg-s3dis
```

Note that if you trained on an area other than 1, you must specify the
`area` flag for `merge-s3dis-predictions` as follows:

```
$ guild run merge-s3dis-predictions area=N
```

where `N` is the area used in `train-seg-s3dis`.

### Segmentation on Semantic3D

System requirements:

- 7z program for unpacking dataset files

First, download and prepare the Semantic3D data:

```
$ guild run prepare-semantic3d-data
```

If `prepare-semantic3d-data` fails during download, you can restart
the operation to begin where you left off using `guild run --restart
RUN_ID` where `RUN_ID` is the ID of the prepare run. Use `guild runs`
for a list of run IDs.


### Segmentation on ScanNet

ScanNet is not currently supported in [guild.yml](guild.yml).

## Testing

This project supports a number of tests. To run the full test suite,
run:

```
$ guild test
```

## To Do

The following is not currently supported in [guild.yml](guild.yml):

- ScanNet
- Hightest accuracy rather than last
