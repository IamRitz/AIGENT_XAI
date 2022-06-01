This repository contains implementations for two problems statements:

<h3>1. Minimal Modifications in DNNs:</h3>
This is a white box approach to correct faults in the network. The implementation can be found in following files:
1. NetworkCorrection/Gurobi/minimalModificationLayers.py: This file contains code to find minimal modification across layers in ACAS-Xu Network. The details of this network can be found in NetworkCorrection/Report-V1 in section 1.3.
2. NetworkCorrection/Gurobi/minimalModificationLayersOriginal.py: This file contains code to find minimal modification only in the last layer to correct a property in the ACAS-Xu Network. This code replicates the original work which has been done in this area. The implementations corresponding to sections 1.2.1 and 1.2.2 are in this file. To switch between these implementations, follow the comments in the file.
3. NetworkCorrection/Gurobi/applyCorrection.py: Call the main method in this file to generate a minimal modification(across layers), apply it to the original network and test it on the inputs.

<h3> 2. Adversarial Image generation</h3>
This is a black box approach which generates adversarial examples for a Network. These adversarial samples can be used to retrain the network and improve it's robustness. Implementation is in following files:
1. NetworkCorrection/Gurobi/adversarialExampleMNIST.py: Contains code to find adversarial image for a particular input image on a trained MNIST model.
2. NetworkCorrection/Gurobi/adversarialExampleACASXU.py: Contains code to find adversarial image for a particular input image on a trained ACAS-Xu model.
3. NetworkCorrection/Gurobi/attackMNIST.py: Contains code to find adversarial images for all the images in the input dataset for MNIST model. It generates an adversarial dataset and stores it in: NetworkCorrection/Gurobi/MNISTdata/adversarialData.csv.


Supervisors: Subodh Vishnu Sharma and Kumar Madhukar, IIT Delhi.
