# README

This is a Caffe2 operator microbenchmark.

## How to use
If you want to know the intensity in advance, you should add config of operator like we provide.

For example
```sh
python3 Calculator.py --config-file Config_Conv.json
```


```sh
cd [OPERATOR]
```

If you want to get the execution time

Using CPU

```sh
python3 [FILENAME]
```

Using GPU

```sh
python3 [FILENAME] --use-gpu
```

If you want to get the CPU-GPU data transfer time

```sh
nsys profile --trace=cuda python3 [FILENAME] --use-gpu
```

```sh
nsys stats [.qdrep]  // Output file from nsys profile
```