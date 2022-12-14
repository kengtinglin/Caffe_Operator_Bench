# README

This is a Caffe2 operator microbenchmark.

## How to use
### Environment Setup
```sh
pip install -r requirements.txt
```
### Roofline
If you want to know the intensity per batch in advance and draw the Roofline, you should add config of operator like we provide.

Example:
```sh
cd Roofline
python3 Roofline.py --config-file Config/Config_Conv.json
```
### Operator Microbenchmark
```sh
cd [OPERATOR]
```

If you want to get the execution time

- Using CPU

    ```sh
    python3 [FILENAME]
    ```

- Using GPU

    ```sh
    python3 [FILENAME] --use-gpu
    ```

If you want to get the CPU-GPU data transfer time

```sh
nsys profile --trace=cuda python3 [FILENAME] --use-gpu
nsys stats [.qdrep]  // Output file from nsys profile
```
## Supported Operators
- Convolution
- Fully Connected
- SparseLengthsSum
