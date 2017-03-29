# AI_Art

To run the code: need an installation of Caffe with built pycaffe libraries, as well as the python libraries numpy, scipy and PIL. For instructions on how to install Caffe and pycaffe, refer to the installation guide [here](http://caffe.berkeleyvision.org/installation.html). Before running the ipython notebooks, you'll also need to download the [bvlc_googlenet model](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet), and insert the path of the pycaffe installation into ```pycaffe_path``` and the model path to the googlenet model into ```model_path```.

This code was based on the [deepdream code](https://github.com/google/deepdream) shared by Google, as well as (https://github.com/kylemcdonald/deepdream/blob/master/dream.ipynb) by Kyle McDonald and Auduno's article and [code](https://github.com/auduno/deepdraw) on visualizations with GoogleNet. The idea of using bilateral filtering comes from Mike Tyka.

https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a - image net class id to class
