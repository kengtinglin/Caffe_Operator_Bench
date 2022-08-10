# README

This is a Caffe2 operator microbenchmark.

### How to use
`cd [OPERATOR]`

If you want to get the execution time

Using CPU

`python3 [FILENAME]`

Using GPU

`python3 [FILENAME] --use-gpu`

If you want to get the GPU data transfer time

`nsys profile --trace=cuda python3 [FILENAME] --use-gpu`

`nsys stats [*.qdrep]`