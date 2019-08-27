# GAN for Synthetic EEG Data

In this project, we explore the application of a generative adversarial network (GAN)
to the production of synthetic EEG data. We examined the benefit of using this synthetic 
data for dataset augmentation, with the goal of increasing classification accuracy of a SVM in a
motor imagery interaction task. 

We implement and train a GAN, and run three experiments to evaluate the potential increase in
classification accuracy of a logistic regression classifier when trained on an augmented training
set rather than a training set consisting of only real EEG data. 

### Background
In the growing field of research for brain computer interfaces (BCI), EEG data is often measured
and classified with various machine learning techniques in order to creative adaptive
interfaces based around brain signals. One common problem, which comes from the time consuming
nature of collecting EEG data, is a shortage of data available to train models. We explore whether
dataset augmentation with GAN could help mitigate this challenge in applying classification algorithms
to EEG datasets in the field of BCI.

For more specifics on our model, methods, and results, please refer to our report which can be found
in 'report.pdf'.
