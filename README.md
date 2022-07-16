<h3>Install the following packages:</h3>
<ol>
  <li>tensorflow==2.8.0</li>
  <li>gurobipy</li>
  <li>keras==2.9.0</li>
  <li>scipy</li>
  <li>pillow</li>
  <li>six</li>
</ol>

These packages can also be installed using the requirements.txt file.
Make sure the python version is python 3.7 and above. If the python version is below 3.7, use the following to install and change the version:

    sudo apt install python3.8
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
    sudo update-alternatives  --set python3 /usr/bin/python3.8

This repository contains implementations for the paper submitted to AAAI'23. 


<ol>
  <li><h3>python3 attack.py:</h3> Attacks MNIST images and calculate Pielou measure: 
      </br>This will perform attack on 500 images from the MNIST dataset and calculate the output impartiality score and the average number of pixels modified.</li>
  <li>
</ol>
<h3>Supervisors: Subodh Vishnu Sharma and Kumar Madhukar, IIT Delhi.</h3>

Link for datasets used and adversarial dataset generated: [Datasets.](https://drive.google.com/drive/folders/1ZUmj_j5fvPiEDmmdNx2QT6gpzIgGubt8?usp=sharing)

Comparitive analysis of different attacks can be found at: [Comparitive analysis.](https://docs.google.com/spreadsheets/d/1sjgXiB_wOy-A0DOaev2TLBRVxU05cY_G9p3KHFTPoRo/edit#gid=0)
