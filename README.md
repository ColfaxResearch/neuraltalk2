
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

the eval script:

```bash
$ th eval.lua -model /path/to/model -image_folder /path/to/image/directory -num_images 512 -batch_size 64 -gpuid -1 -dump_images 0
```

This tells the `eval` script to run up to 512 images from the given folder, in batches.

or just run the script:
```bash
sh cpu_benchmark.sh
```
### License

BSD License.

### Acknowledgements

Many thanks to the original authors of this project Andrej Karpathy and Justin Johnson for making this available.
