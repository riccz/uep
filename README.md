# Unequal error protection

## Dependencies
* GCC >= 6.3
* CMake >= 3.2
* Boost libraries >= 1.58
* Protobuf library and compiler >= 3.0
* Python >= 3.5 with development headers

On Debian Stretch the required packages can be installed by running
`sudo apt-get install build-essential cmake libboost-all-dev
protobuf-compiler libprotobuf-dev libpthread-stubs0-dev python3-dev`

## Build
This project uses the [CMake](https://cmake.org/) build system.  To
compile the project inside a subdirectory named `build` run, in the
project's main directory
* `mkdir -p build && cd build`
* `cmake -DCMAKE_BUILD_TYPE=Release ..`
* `make`

The compiled binaries will be in the `build/bin` subdirectory.

## Test
The automated unit tests can be run with `make test` from the `build`
directory.

## Run
There are two main interfaces to the simulated system:
* _`mppy`_ is a Python module that uses directly the
  uep::message_passing class to simulate the decoding
  process. This is used by the scripts run_uep_{iid|markov|time}.py
  to build the PER and time plots, that can be viewed by the
  corresponding plot_uep_{iid|markov|time}.py scripts.
* _`server` and `client`_ are two programs that actually transmit
  a video across a network. The video needs to be pre- and post-
  processed using the scripts in the `h264_scripts` directory.
  This programs and the required scripts are tied together by
  run_ber_iid.py and run_markov2_vs_k.py which setup and run a
  server and client on the sam machine.
