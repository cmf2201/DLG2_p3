Training or Testing Network for Adversarial Patch Attack on Monocular Depth Estimation Networks
====

When you use the codes for research purpose, please include citation to the following paper in your written documents and presentations.

K. Yamanaka, R. Matsumoto, K. Takahashi, and T. Fujii,
``Adverasarial Patch Attacks on Monocular Depth Estimation Networks,''
IEEE Access, Vol. 8, pp. 179094--179104, 2020.


-----------------------------------------
## Citation
-----------------------------------------
The zip archive includes the distribution codes of the following papers.
The downloaded data are stored in "repository" directory.

X. Guo, H. Li, S. Yi, J. Ren, and X. Wang,
``Learning monocular depth by distilling cross-domain stereo networks,''
Proceedings of the European Conference on Computer Vision (ECCV), pp. 484--500, 2018.

Source Code (Github)
https://github.com/xy-guo/Learning-Monocular-Depth-by-Stereo/archive/master.tar.gz

Network weight (GoogleDrive)
https://drive.google.com/uc?export=download&id=1Jxb0A3FaNZJQPFR5p8HIqpAS3pvIBZFL


-----------------------------------------
## Content
-----------------------------------------
* Training, testing and visualization codes used in the paper.

* Trained adversarial patches (Fig.4)
We put these patches in "Src/trained_path" directory.
P_{n}^{1}: "circle_patch_distill_near"
P_{f}^{1}: "circle_patch_distill_far"
P_{n}^{2}: "circle_patch_bts_near"
P_{f}^{2}: "circle_patch_bts_far"
P_{n}^{*}: "circle_patch_distill_bts_near"
P_{f}^{*}: "circle_patch_distill_bts_far"


P_{n}^{1} can fool depth estimates by Guo's network to being very close on the region in which our adversarial patch is located, regardless of the actual depth.
On the other hand, P_{f}^{1} can fool that to being very far.

P_{n}^{2} and P_{f}^{2} can fool depth estimates by Lee's network which is pytorch version, trained on the KITTI dataset.
The source code is available in "https://github.com/cogaplex-bts/bts".

P_{n}^{*} and P_{f}^{*} can fool depth estimate by both Guo's network and Lee's network.


* Folders structure
.
|-- Dst
|   |-- checkpoints           # will store training results by 'train.py'
|   |-- test_result           # will store testing results by 'test.py'
|   `-- visualization_result  # will store visualization results by 'visualize.py'
|-- Src
|   |-- input_img             # We put dummy training data here
|   |-- list                  # Text files containing information necessary for learning
|   `-- trained_patches       # Adversarial patches in papers
|-- models
|-- repository                # Source codes and weight published by Guo et al.
`-- utils


-----------------------------------------
## Requirement
-----------------------------------------
The included codes require Python (version 3.6.9 or higher), CUDA (version 9.2 or higher) and cuDNN (version 7.1.4 or higher).
The required python packages are pytorch (version 1.1.0 or higher), torchvision (version 0.4.0 or higher), opencv-python, matplotlib, numpy, scipy and torchgeometry.
We have tested the software on Ubuntu 18.04 LTS and in the docker-environment (Ubuntu 18.04 LTS) with Nvidia Geforce GTX 1080 ti.
You need at least 10 GB of GPU memory to run this program.
We put the dockerfile used for testing in the current directory (./Dockerfile).

### Warning
If you are using a higher version of pytorch more than 1.1.0, you may get the following error.
In that case, please follow the error message and modify the source code of torchgeometry.
"File "/usr/local/lib/python3.6/dist-packages/torchgeometry/core/imgwarp.py", line 258, in get_perspective_transform"
   258: X, LU = torch.gesv(b, A)
=> 258: X, LU = torch.solve(b, A)


-----------------------------------------
## Install
-----------------------------------------
To learn, test, and visualize, you must first execute the following command to complete the necessary steps.

$ bash ./install.sh


-----------------------------------------
## Usage
-----------------------------------------
The training, testing, and visualization codes described below can all be executed without any arguments.
In that case, the test data we have prepared will be used.
If you use your own data, you need to execute them with appropriate arguments.
The default input/output is written at the top of the program's text output.


### TRAINING
We locate the sample training data in "Src/input_img" directory.
What is located in this directory is a dummy dataset that is different from the one we actually used in the above paper.

To train an adversarial patch, execute the following command.

$ python3 train.py

If you want to train adversarial patches with KITTI dataset used in the above paper, please download the dataset by executing the following command.
$ DATASET_PATH=./Src/input_img/kitti  # or any other directory
$ mkdir -p $DATASET_PATH
$ wget -i ./repository/Learning-Monocular-Depth-by-Stereo-master/scripts/kitti_raw_dataset_files.txt -P $DATASET_PATH
$ cd $DATASET_PATH
$ unzip '*.zip'

To train using them, execute the following command.
By executing this command, you can start learning the patches P_{n}^{1} shown in Fig. 4.
An epoch takes about 3 hours on a GTX1080 machine.

$ python3 train.py --data_root ./Src/input_img/kitti/ --train_list ./Src/list/eigen_train_list.txt


### TEST
We include our trained adversarial patches in "Src/trained_patch", and a sample dataset of 4 scenes in "input_img/test_scenes*.jpg".
To test these patches, execute the following command.
It takes about 0.6 second per image, including loading the image, attacking, estimating, and saving the estimate.

$ python3 test.py

This test command outputs the results corresponding to the two left-hand columns of Fig. 3(a) in the paper.
The estimated results will be generated in "Dst/test_result" directory.

If you want to test with your own image, please add the path to the image after "img_path" argument and execute the following command.

$ python3 test.py --img_path [PATH_TO_TEST_IMAGE]

If you want to use the adversarial patches you learned for testing, add the path to the patch and mask after "patch_path" and "mask_path" argument and execute the following command.

$ python3 test.py --patch_path [PATH_TO_TRAINED_PATCH] --mask_path [PATH_TO_MASK]

The trained adversarial patches and masks are stored in "Dst/checkpoints/default" directory by default.


### VISUALIZE
To perform the two types of visualization in the paper, execute the following command.
It takes about an hour to output visualization results on a GTX1080 machine.

$ python3 visualize.py

This visualization command outputs the results corresponding to Figs. 9 and 10 in the paper.
The estimated results will be generated in "Dst/visualize_result" directory.
Note that the visualization of the "potentially affected region" in section 5.A is exported as an '.npy' file.
If you want to create an image like the last column in Fig. 9, use "Dst/visualization_result/show_potentially affected region.py".

$ cd Dst/visualization_result
$ python3 show_potentially_affected_region.py


If you want to visualize your own image, please add the path to the image after "img_path" argument and execute the following command.

$ python3 visualize.py --img_path [PATH_TO_TEST_IMAGE]

As with testing, if you want to use your own learned patches for visualization, add the path to the patch and mask after "patch_path" and "mask_path" argument and execute the following command.

$ python3 visualize.py --patch_path [PATH_TO_TRAINED_PATCH] --mask_path [PATH_TO_MASK]


-----------------------------------------
## How to Attack Monocular Depth Estimation Networks in the Real World
-----------------------------------------
To perform an attack for Guo et al.'s method in the real world, you need to print the patch you created.
Adversarial patches should be printed on a non-glossy material so that the pattern of the patch is not erased by reflection.
Our adversarial patches P_{n}^{1} and P_{f}^{1} were printed on matte paper and attached to a 7 mm thick panel.
The printed P_{n}^{1} and P_{f}^{1} had diameters of 32 and 64 cm, respectively.
In addition, depending on the training conditions, the image should be taken so that adversarial patches occupy at least 2.4% of the total pixels in the image.

After placing the printed adversarial patches in the real world, you can photograph the patches so that they are included in the image.
Saves the image to a desired directory.
By default, the test program resizes the width of the input image to 512 pixels and the height to 256 pixels (to match the input of Guo et al.'s network).
Therefore, it is better to take the image with a 2:1 width-to-height ratio or to edit the image for better results.

Afterwards, add "no_overwrite" argument and the path to the image after "img_path" argument, then execute the following command.

$ python3 test.py --img_path [PATH_TO_PHOTOGRAPHED_IMAGE] --no_overwrite

Executing 'test.py'  with the argument 'no_overwrite' skips the process of digitally overwriting the adversarial patches and simply estimates the depth of the input image using the Guo et al's method.


If you want to attack Lee et al.'s method, print patches P_{n}^{2} and P_{f}^{2} in the same way as above and take a picture.
Then estimate the depth of the attacked image using the source code published by Lee et al.
In this case, note that you should use Lee et al.'s network which is pytorch version, trained on the KITTI dataset.
The source code is available in "https://github.com/cogaplex-bts/bts".


-----------------------------------------
## License
-----------------------------------------
The MIT License (MIT)

Copyright (c) 2020 Koichiro Yamanaka at Fujii Lab. of Nagoya University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.


THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
