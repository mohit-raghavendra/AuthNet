# Authnet
![Header](https://raw.githubusercontent.com/Mohit-Mithra/AuthNet/master/pipeline_authnet.jpg "Header")
<br><br>
This is the code repository for the paper:

<b>AUTHNET - A deep learning based authentication mechanism using temporal facial feature movements</b> <br>

Face recognition systems are typically based on a still image of a face picture and comparing it with the previous data to make predictions. The aim of the project is to develop a face recognition model that can strengthen the existing system by making use of facial movement patterns. It involves training the system on videos of the user uttering a predetermined password. Once that is done, the system can identify if a video of a person uttering a word is the correct person saying the correct password or not. 

This can be used as a facial recognition system, where the system “recognises” the right person uttering the right password.

<b>AuthNet was accepted for oral presentation at AAAI-2021 for the Student Abstract and Poster Program. </b><br><br>
Find our paper on arxiv: <https://arxiv.org/abs/2012.02515> and on the AAAI proceedings: <https://ojs.aaai.org/index.php/AAAI/article/view/17933> <br>
<br><br>

The implementation is completely on jupyter notebooks. Hence, it is advised to download anaconda or miniconda packages to run this code. <br><br>

------
## Prerequisites:
* OpenCV for python
* Numpy
* Matplotlib
* Scikit-learn
* Tensorflow

----------

### Clone the repo:<br>
```
git clone https://github.com/Mohit-Mithra/AuthNet.git 
cd AuthNet\Cerberus
```
--------

### Follow these steps to train the model for one speaker and test it with various videos: <br><br>
1. We have used a keras version of the pretrained VGGFace MatLab model available [here](https://www.vlfeat.org/matconvnet/pretrained/). This model is needed for training the model as well as using the system. Add it to the 'Utilities' folder. <br> You can convert the model to it's Keras version yourself by running [this](https://github.com/Mohit-Mithra/AuthNet/blob/master/Cerberus/Library/Bottle%20Cap.ipynb).
 
3. Inside the 'Cerberus' directory, create a new folder called 'Stored_Negative' and inside that folder add this [file](https://drive.google.com/file/d/1PB1X1IqqNIfzplwtDGKCQ9_JHjBHFtiF/view?usp=sharing). <br>

For adding the data, follow these steps: <br>
  * Inside 'Cerberus' directory, create a 'videos' folder and the videos for the utterances. Note: This model has been built for 5 utterances per speaker. <br>
  * Also, create an empty 'photos' folder with the following empty subfolders: <br>
        -> utterance 1<br>
        -> utterance 2 <br>
        -> ... <br>
        -> utterance 5<br>
        
     This folder will be populated with the sliced images of the video which will then be fed into VGGFace.<br><br>
3. Now for testing, create a folder called 'test' inside 'Cerberus'. Then, inside the 'test' folder follow the same steps listed above. <br><br>

4. Finally,run Cerberus/Cerebrus_training.ipynb. This will create the trained AuthNet model for that speaker. <br>
<i>Note: This version does not use Haar Cascade filters as videos taken from smartphones do not need any cropping.</i><br>
5. You can run Cerberus/Cerberus_testing.ipynb with different videos added in the 'test' folder to obtain predictions from the model. 

----------
<br>
Cite us:

```
@article{Raghavendra_Omprakash_B R_2021, 
title={AuthNet: A Deep Learning Based Authentication Mechanism Using Temporal Facial Feature Movements (Student Abstract)}, volume={35}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/17933}, 
abstractNote={Deep learning algorithms are widely used to extend modern biometric authentication mechanisms in resource-constrained environments like smartphones, providing ease-of-use and user comfort, while maintaining a non-invasive nature. In this paper, an alternative is proposed, that uses both facial recognition and the unique movements of that particular face while uttering a password. The proposed model is language independent, the password doesn’t necessarily need to be a set of meaningful words or numbers, and also, is a contact-less system. When evaluated on the standard MIRACL-VC1 dataset, the proposed model achieved a testing accuracy of 98.1%, underscoring its effectiveness.}, 
number={18}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Raghavendra, Mohit and Omprakash, Pravan and B R, Mukesh}, 
year={2021}, 
month={May}, 
pages={15873-15874} 
}

```
