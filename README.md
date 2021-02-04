# Authnet
![Header](https://raw.githubusercontent.com/Mohit-Mithra/AuthNet/master/pipeline_authnet.jpg "Header")
This is the code repository for the paper:

<b>AUTHNET - A deep learning based authentication mechanism using temporal facial feature movements</b> <br>

Face recognition systems are typically based on a still image of a face picture and comparing it with the previous data to make predictions. The aim of the project is to develop a face recognition model that can strengthen the existing system by making use of facial movement patterns. It involves training the system on videos of the user uttering a predetermined password. Once that is done, the system can identify if a video of a person uttering a word is the correct person saying the correct password or not. 

This can be used as a facial recognition system, where the system “recognises” the right person uttering the right password.

<b>AuthNet was accepted for oral presentation at AAAI-2021 for the Student Abstract and Poster Program. </b><br><br>
Find our paper on arxiv: <https://arxiv.org/abs/2012.02515> <br>
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
1. We have created a keras version of VGGFace available [here](https://drive.google.com/file/d/1cgNbT4UOGyEiAcB64vqwkhNtp-XCsL3u/view?usp=sharing). This model is needed for training the model as well as using the system. Add it to the 'Utilities' folder. <br>
2. Inside the 'Cerberus' directory, create a new folder called 'Stored_Negative' and inside that folder add this [file](https://drive.google.com/file/d/1PB1X1IqqNIfzplwtDGKCQ9_JHjBHFtiF/view?usp=sharing). <br>

For adding the data, follow these steps: <br>
  * Inside 'Cerberus' directory, create a 'videos' folder and the videos for the utterances. Note: This model has been built for 5 utterances per speaker. <br>
  * Also, create an empty 'photos' folder with the following empty subfolders: <br>
        -> utterance 1<br>
        -> utterance 2 <br>
        -> ... <br>
        -> utterance 5<br>
        
     This folder will be populated with the sliced images of the video which will then be fed into VGGFace.<br><br>
3. Now for testing, create a folder called 'test' inside 'Cerberus'. Then, inside the 'test' folder follow the same steps listed above. <br><br>

4. Finally,run Cerberus/Cerebrus_training.ipynb. This will create the trained AuthNet model for that speaker. <br><br>
5. You can run Cerberus/Cerberus_testing.ipynb with different videos added in the 'test' folder to obtain predictions from the model. 

----------
<br>
Cite us:

```
@misc{raghavendra2020authnet,
  title={AuthNet: A Deep Learning based Authentication Mechanism using Temporal Facial Feature Movements},
  author={Mohit Raghavendra and Pravan Omprakash and B R Mukesh and Sowmya Kamath},
  year={2020},
  publisher={2012.02515},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
}
```
