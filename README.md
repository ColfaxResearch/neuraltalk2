
# NeuralTalk2

This is used as a benchmark for the Colfax implementation of Torch. It is intended to run well on Xeon processors or Xeon Phi processors.
To reproduce the results from our publication at [Colfax Research](http://colfaxresearch.com/isc16-neuraltalk/), do the following:
Download and install torch:
```bash
$ git clone https://github.com/ColfaxResearch/Torch-distro.git ~/torch --recursive
$ cd ~/torch
$ ./install-deps
$ ./install.sh      # and enter "yes" at the end to modify your bashrc
$ source ~/.bashrc
$ yum install h5py hdf5-devel
$ cd && git clone https://github.com/deepmind/torch-hdf5.git
$ cd torch-hdf5
$ luarocks make
$ yum install protobuf-devel protobuf-compiler
$ luarocks install loadcaffe
```
Download a pretrained checkpoint from the original author of NeuralTalk2:
The pretrained checkpoint can be downloaded here: [pretrained checkpoint link](http://cs.stanford.edu/people/karpathy/neuraltalk2/checkpoint_v1.zip) (600MB). 

Download some pictures from COCO or where else you'd like:
[COCO Download](http://mscoco.org/dataset/#download)

watch a fun video with our results while the download is going:
[Video Link](https://www.youtube.com/watch?v=tRY2WYfen3g)

one useful timer module used:
```bash
cd ~/neuraltalk2/
C_INCLUDE_PATH=~/torch/install/include gcc -Wall -shared -fPIC -o walltime.so -llua-5.1 walltime.c
```
the eval script:

```bash
$ th eval.lua -model /path/to/model -image_folder /path/to/image/directory -num_images 512 -batch_size 64 -gpuid -1 -dump_images 0
```

This tells the `eval` script to run up to 512 images from the given folder, in batches.

or just run the script:
```bash
sh cpu_benchmark.sh
```

explanations for a few points on the graph:

The rockspec files for the torch7 and thnn packages have been changed to use the Intel tools and libraries.

User code change happened in neuraltalk2/misc/LanguageModel, where a sort is replaced with topk

The improved parallel strategy refers to running the network like this:
```bash
NPR=16 # Number of concurrent processes
NTH=16 # Threads per process
OFS=0
PIDS=""
rm offset*
for ((i=0; $i<$NPR; i++)); do
KMP_AFFINITY=compact,granularity=fine,0,$OFS   OMP_NUM_THREADS=$NTH th
eval.lua -model model_id1-501-1448236541.t7_cpu.t7 -image_folder ~/coco_images/  -num_images 72 -batch_size 1 -gpuid -1 -dump_images 0 > offset_${OFS}.txt & PIDS="$PIDS $!"
let OFS=OFS+NTH
done
echo $PIDS
wait $PIDS
```
The reason for doing this is that the LSTM portion of the network does not benefit from using more than a few cores, while the CNN portion's performance scales linearly with the number of cores used.
Using multiple processes with a smaller number of cores each allows for a better utilitization over all.

MCDRAM is a feature of the second generation Xeon Phi processors. It was set to be used as Cache in the BIOS.

### License

BSD License.

### Acknowledgements

Many thanks to the original authors of this project Andrej Karpathy and Justin Johnson for making this available.
