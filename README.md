# annf
Software repository for my masters thesis about Data-Parallel Coherency Sensitive Hashing
## Dependencies

The annf package has been tested under Python 3.6.8. The required Python dependencies are:

- numpy==1.19.5
- pyopencl==2021.1.6

A working version of Futhark needs to be installed. Latest Futhark version annf has been tested on is 0.19.0.
[OpenCL](https://www.khronos.org/opencl) needs to be available.

## Setup
The repository can be used by navigating to csh-fut, installing the necessary packages:

I recommend using virtualenv for installing the dependencies, for example:

  `$ virtualenv -p python3 annfield`

  `$ source annfield/bin/activate`

  `$ pip install -r requirements.txt`
 
An example use of the program can be run by:

`$ futhark pyopencl --library csh-fut/csh.fut`

`$ python csh-fut/example.py`