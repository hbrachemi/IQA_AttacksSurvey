# IQA_AttacksSurvey

# Contents
1. [Abstract](#Abstract)
2. [General Attack Outline](#General-Attack-Outline)
3. [Examples](#Examples) 
4. [Usage](#Usage)
5. [Contact](#Contact)


## Abstract
The rapid development of Deep Learning (DL) and,
more specifically, Convolutional Neural Networks (CNNs) has
achieved high accuracy over the past decade, becoming the
standard approach in computer vision in a short time. However,
recent studies have discovered that CNNs are vulnerable to
adversarial attacks in image classification tasks. While most
studies have focused on DL models for image classification, only
a few works have addressed this issue in the context of Image
Quality Assessment (IQA). This paper investigates the robustness
of different CNN models against adversarial attacks when used
for an IQA task. We propose an adaptation of state-of-the-
art image classification attacks in both targeted and untargeted
modes for an IQA regression task. We also analyze the correlation
between the perturbation’s visibility and the attack’s success.
Our experimental results show that DL-based IQA methods
are vulnerable to such attacks, with a significant decrease in
correlation scores when subjected to adversarial perturbations.
Consequently, the development of countermeasures against such
attacks is essential for improving the reliability and accuracy of
DL-based IQA models.
## General Attack Outline
![](https://github.com/hbrachemi/IQA_AttacksSurvey/blob/master/schema.png)
## Examples
![](https://github.com/hbrachemi/IQA_AttacksSurvey/blob/master/examples.png)
## Usage
[AttacksGenerationOnDataset.ipynb](https://github.com/hbrachemi/IQA_AttacksSurvey/blob/master/AttacksGenerationOnDataset.ipynb) provides a guideline on how to launch the attack on a dataset.

1. Set the path to the target dataset's repository
```db = "../Databases/tid2013/"```
2. Define the maximum scale of the ground truth quality score:
```scale = int(input())```
3. Define the victim model and load its weights
```model = initialize_model('inception',False,True)```
 ```  weights_path = '../pretrained/iqaModel_tid_inception.pth'```
4. Define the parametters of the attack
```iterations = [10]```
```epsilons = [0.001,0.01,0.1]```
```attacks = ["bim","pgd","fgm"]```
```losses = ['mse(y_tielda,y)']```
 ## Contact
Hanene F.Z Brachemi Meftah , `hanene.brachemi@insa-rennes.fr`

   
