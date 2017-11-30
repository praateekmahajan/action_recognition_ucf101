ction Recognition for UCF 101 

Final project for my Computer Vision class, to recognize class of a given video based. So far limited work has been done in using temporal features for feature detection, my aim is to **either** compare the latest state of the art tools for action recognition or to implement a vanilla solution. 

The dataset used here is [UCF 101](http://crcv.ucf.edu/data/UCF101.php), you can read more about it on the website. The first train/test splits are used from the three train/test splits provided.
## Workflow

### Step one

  - Load videos from training file trainlist01.txt) convert it to frames.
  - Extracting two frames per second.
  - Save it in a folder named UCF101_images, and create a file named image_train1.txt
  - Run a CNN over these images with their labels.
 
### Step two
  - After training the CNN, pass the last layer of CNN to LSTM
 



## Installation

This requires python2.7 and pytorch.

Install the dependencies and devDependencies and start the server.

```sh
cd action_recognition_ucf101
conda create --name action
source activate action
pip install scikit-video
conda install pytorch torchvision -c soumith
conda install jupyter
```
