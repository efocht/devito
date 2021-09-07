


## Prerequisites

You need to have `llvm-ve-rv-1.9.0` installed. The example below is for CentOS 7. Replace
`el7` by `el8` for CentOS 8.
```
yum install https://sx-aurora.com/repos/llvm/x86_64/llvm-ve-rv-1.9.0-1.9.0-1.el7.x86_64.rpm
```

For llvm-ve-rv running native OpenMP Target offloading you will need a
`veorun` that is linked without the NEC OpenMP. Otherwise you'll
notice NEC's OpenMP interfering with the llvm-ve OpenMP, thus starting
16 threads on 8 cores.
```
touch _dummy.c
/opt/nec/ve/bin/ncc -o _dummy.o -c _dummy.c
/opt/nec/ve/bin/mk_veorun_static -o veorun-nomp _dummy.o
rm -f _dummy.o _dummy.c

export VEORUN_BIN=$(pwd)/veorun-nomp
```

Install python3 and pip3, including the development package.
```
yum -y install python3 python3-dvel python3-pip
```

Set the environment variables for using LLVM-VE:
```
. /usr/local/ve/llvm-ve-rv-1.9.0/bin/llvmvars.sh
```


## Checkout Devito and install locally

```
git clone https://github.com/efocht/devito.git
cd devito
git checkout nec-ve
cd ..

pip3 install --user file://$(pwd)/devito
```


## Run an example

Common environment variable settings for Devito:
```
export DEVITO_ARCH=clang
export DEVITO_PLATFORM=necveX
export DEVITO_LANGUAGE=openmp
export DEVITO_DEVELOP=1
export DEVITO_LOGGING=DEBUG
```

Copy an example:
```
cp -p devito/examples/seismic/acoustic/acoustic_example.py .
```


### Running with LLVM-VE native OpenMP Target

For this case the offloaded VE code is compiled with llvm-ve-rv.
```
export VEORUN_BIN=$(pwd)/veorun-nomp
unset VE_OMPT_SOTOC

python3 acoustic_example.py
```

### Running with NCC compiled OpenMP Target offloaded code

In this case the VE code is compiled with ncc.
```
unset VEORUN_BIN
export VE_OMPT_SOTOC=y

python3 acoustic_example.py
```


## Tuning and Options

### Note

The generated code is very much similar to what a GPU would get with
OpenACC or OpenMP Target. The code is not optimized and has less
`#pragma omp simd` directives than some CPU codes. We must work on
optimizing the compile chain of devito to produce more appropriate
code for the SX-Aurora. Also VE specific stages are imaginable, which
insert NC specific directives for the SOTOC style omp target
execution.



The compile options for clang are defined in `devito/arch/compiler.py`
in a block defined for the "platform" **NECVEX**.

For SOTOC (ncc compiled offloaded code) additional compile options
aimed at ncc need to be passed to llvm with `-Xopenmp-target ...`.
