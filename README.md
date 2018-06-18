# Replicate MURA baseline and playing around

Reference https://stanfordmlgroup.github.io/competitions/mura/

# How to run the script

1) pip install virtualenv

2) git clone git@github.com:DeepMachineLearning/mura-team1.git

   Or

   git clone https://github.com/DeepMachineLearning/mura-team1.git

3) cd mura-team1

4)virtualenv env --python=python3.6

5)source env/bin/activate

Note: Need to do the above command to use the environment before run script "python mnist_vgg.py train -e 5 -b 64" each time. Otherwise, the command "python mnist_vgg.py train -e 5 -b 64" will not able to find mnist library and others because mnist and other libraries are installed at current env folder.

6) Use "pip list" to see what libraries are installed

7) Install libraries

pip install -r requirements.txt

8) git fetch origin vgg-xiu:vgg-xiu

9) git checkout vgg-xiu

10) (Opitional) Add TkAgg in matplotlibrc

vim ~/.matplotlib/matplotlibrc

add backend:TkAgg

11) Download mnist library from http://yann.lecun.com/exdb/mnist/ and copy the exacted file to dataset folder which is under mura-team1 folder

12) Download MURA from https://stanfordmlgroup.github.io/competitions/mura/. Register first and get an email include the MURA data set.

13 run script to get result

python mnist_vgg.py train -e 5 -b 64