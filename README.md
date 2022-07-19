<h3>Install the following packages:</h3>
<ol>
  <li>tensorflow==2.8.0</li>
  <li>gurobipy</li>
  Our tool requires a licensed version of Gurobi. Please follow the instructions on: https://www.gurobi.com/free-trial/ to install a licensed version using an academic license or a professional license.
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
<h4>AttackMNIST and AttackCIFAR have similar folder structures. To launch attack on the datasets and calculate various norms, run the following commands:</h4>
<ol>
  <li><h5>python3 launchAttack.py:</h5> Attacks original images and calculate Pielou measure: 
      </br>This will perform attack on 500 images from the original dataset and calculates the output impartiality score and the average number of pixels modified. Pielou Measure should be closer to 1.</li>
  <li><h5>python3 fid.py:</h5> Calculates FID(naturalness score) for the adversarial images generated. FID should be closer to 0.</li>
  <li><h5>python3 calculateNorms.py:</h5> Calculates average L2 and L-inf norms for the adversarial images generated.</li>
  <li><h5>python3 generateImages.py:</h5> Generates adversarial Images and saves them to specified folder.</li>
</ol>
<h3>Supervisors: Subodh Vishnu Sharma and Kumar Madhukar, IIT Delhi.</h3>

Link for complete datasets used and adversarial dataset generated: [Datasets.](https://drive.google.com/drive/folders/1ZUmj_j5fvPiEDmmdNx2QT6gpzIgGubt8?usp=sharing)

<!-- Comparitive analysis of different attacks can be found at: [Comparitive analysis.](https://docs.google.com/spreadsheets/d/1sjgXiB_wOy-A0DOaev2TLBRVxU05cY_G9p3KHFTPoRo/edit#gid=0) -->
