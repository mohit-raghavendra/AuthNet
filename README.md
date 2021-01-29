# Authnet

This is the code repository for the paper:

AUTHNET - A deep learning based authentication mechanism using temporal facial feature movements <br>

Face recognition systems are typically based on a still image of a face picture and comparing it with the previous data to make predictions. The aim of the project is to develop a face recognition model that can strengthen the existing system by making use of facial movement patterns. It involves training the system on videos of the user uttering a predetermined password. Once that is done, the system can identify if a video of a person uttering a word is the correct person saying the correct password or not. 

This can be used as a facial recognition system, where the system “recognises” the right person uttering the right password.

AuthNet was accepted for oral presentation at AAAI-2021 for the Student Abstract and Poster Program. 
Find our paper on arxiv: <https://arxiv.org/abs/2012.02515> <br>
<br>

You will need need to add a 'Utilities' folder structure and add the following files: <br>
-> white.jpg (white image for padding) <br>
-> Download the VGGface.h5 file from here ([VGGFace](https://drive.google.com/file/d/1cgNbT4UOGyEiAcB64vqwkhNtp-XCsL3u/view?usp=sharing)) <br>
<br>
-> Create a 'videos' folder that contains the videos for the utterances </br>
-> Create a 'images' folder with following structure: </br>
   -> images </br>
   
        -> utterance 1 </br>
        -> utterance 2 </br>
        -> ... </br>
        -> utterance 5 </br>
-> Run jupyternotebooks/Cerebrus.ipynb passing in the paths for the 'video' and 'image' folders, at the specified lines.


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
