
# NeuralTalk2

This is used as a benchmark for the Colfax implementation of Torch. It is intended to run well on Xeon processors or Xeon Phi processors.

Recurrent Neural Network captions your images. Now much faster and better than the original [NeuralTalk](https://github.com/karpathy/neuraltalk). Compared to the original NeuralTalk this implementation is **batched, uses Torch, runs on a Xeon or Xeon Phi, and supports CNN finetuning**. All of these together result in quite a large increase in training speed for the Language Model (~100x), but overall not as much because we also have to forward a VGGNet. However, overall very good models can be trained in 2-3 days, and they show a much better performance.

This is an early code release that works great but is slightly hastily released and probably requires some code reading of inline comments (which I tried to be quite good with in general). I will be improving it over time but wanted to push the code out there because I promised it to too many people.

![teaser results](https://raw.github.com/karpathy/neuraltalk2/master/vis/teaser.jpeg)

You can find a few more example results on the [demo page](http://cs.stanford.edu/people/karpathy/neuraltalk2/demo.html). These results will improve a bit more once the last few bells and whistles are in place (e.g. beam search, ensembling, reranking).

There's also a [fun video](https://vimeo.com/146492001) by [@kcimc](https://twitter.com/kcimc), where he runs a neuraltalk2 pretrained model in real time on his laptop during a walk in Amsterdam.

### Requirements


#### For evaluation only

This code is written in Lua and requires [Torch](http://torch.ch/). If you're on CentOS, installing Torch in your home directory may look something like: 

```bash
$ git clone https://github.com/ColfaxResearch/Torch-distro.git ~/torch --recursive
$ cd ~/torch; 
$ ./install-deps
$ ./install.sh      # and enter "yes" at the end to modify your bashrc
$ source ~/.bashrc
```

See the Torch installation documentation for more details. After Torch is installed we need to get a few more packages using [LuaRocks](https://luarocks.org/) (which already came with the Torch install). In particular:

#### For training

If you'd like to train your models you will need [loadcaffe](https://github.com/szagoruyko/loadcaffe), since we are using the VGGNet. First, make sure you follow their instructions to install `protobuf` and everything else (e.g. `sudo yum install protobuf-devel protobuf-compiler`), and then install via luarocks:

```bash
luarocks install loadcaffe
```

Finally, you will also need to install [torch-hdf5](https://github.com/deepmind/torch-hdf5), and [h5py](http://www.h5py.org/), since we will be using hdf5 files to store the preprocessed data.
```bash
$ yum install h5py hdf5-devel
$ git clone https://github.com/deepmind/torch-hdf5.git
$ cd torch-hdf5
$ luarocks make
```

Phew! Quite a few dependencies, sorry no easy way around it :\

### I just want to caption images

In this case you want to run the evaluation script on a pretrained model checkpoint. 
I trained a decent one on the [MS COCO dataset](http://mscoco.org/) that you can run on your images.
The pretrained checkpoint can be downloaded here: [pretrained checkpoint link](http://cs.stanford.edu/people/karpathy/neuraltalk2/checkpoint_v1.zip) (600MB). It's large because it contains the weights of a finetuned VGGNet. Now place all your images of interest into a folder, e.g. `blah`, and run
the eval script:

```bash
$ th eval.lua -model /path/to/model -image_folder /path/to/image/directory -num_images 10 
```

This tells the `eval` script to run up to 10 images from the given folder. If you have a big GPU you can speed up the evaluation by increasing `batch_size` (default = 1). Use `-num_images -1` to process all images. The eval script will create an `vis.json` file inside the `vis` folder, which can then be visualized with the provided HTML interface:

```bash
$ cd vis
$ python -m SimpleHTTPServer
```

Now visit `localhost:8000` in your browser and you should see your predicted captions.

You can see an [example visualization demo page here](http://cs.stanford.edu/people/karpathy/neuraltalk2/demo.html).

**"I only have CPU"**. Okay, in that case download the [cpu model checkpoint](http://cs.stanford.edu/people/karpathy/neuraltalk2/checkpoint_v1_cpu.zip). Make sure you run the eval script with `-gpuid -1` to tell the script to run on CPU.

**Running on MSCOCO images**. If you train on MSCOCO (see how below), you will have generated preprocessed MSCOCO images, which you can use directly in the eval script. In this case simply leave out the `image_folder` option and the eval script and instead pass in the `input_h5`, `input_json` to your preprocessed files. This will make more sense once you read the section below :)

**Running a live demo**. With OpenCV 3 installed you can caption video stream from camera in real time. Follow the instructions in [torch-opencv](https://github.com/VisionLabs/torch-opencv/wiki/installation) to install it and run `videocaptioning.lua` similar to `eval.lua`. Note that only central crop will be captioned.

### License

BSD License.

### Acknowledgements

Parts of this code were written in collaboration with my labmate [Justin Johnson](http://cs.stanford.edu/people/jcjohns/). 

I'm very grateful for [NVIDIA](https://developer.nvidia.com/deep-learning)'s support in providing GPUs that made this work possible.

I'm also very grateful to the maintainers of Torch for maintaining a wonderful deep learning library.
