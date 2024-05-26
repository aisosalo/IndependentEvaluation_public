# Independent evaluation of a multi-view multi-task convolutional neural network breast cancer screening examination classifier

This repository contains source code for finetuning experiments described in "Independent evaluation of a multi-view multi-task convolutional neural network breast cancer classification model using Finnish mammography screening data".

## Dependencies

### Operating system

* Python 3.6

### Software libraries

* pytorch 1.1.0
* torchvision 0.3.0
* qhoptim 1.1.0
* numpy 1.19.2
* scipy 1.5.2
* pandas 0.22.0
* sklearn 0.24.2
* h5py 2.7.1
* imageio 2.4.1
* opencv-python 3.4.2.17
* tensorboardx 1.4
* tqdm 4.64.1
* termcolor 1.1.0

### NYU multi-view multi-task model and pre-trained weights

* https://github.com/nyukat/breast_cancer_classifier

## License

This software is published under the [AGPLv3 licence](https://github.com/aisosalo/IndependentEvaluation/blob/main/LICENSE).

Licenses for third party components are listed in the [NOTICE](https://github.com/aisosalo/IndependentEvaluation/blob/main/NOTICE) file.

The software has not been certified as a medical device and, therefore, must not be used for diagnostic purposes.

## Author

[Antti Isosalo](https://github.com/aisosalo/), Research Unit of Health Sciences and Technology, University of Oulu, Oulu, Finland.

## Acknowledgements

### Jane and Aatos Erkko Foundation and Technology Industries of Finland Centennial Foundation

Financial support from the Jane and Aatos Erkko Foundation and the Technology Industries of Finland Centennial Foundation is gratefully acknowledged.

### Jenny and Antti Wihuri Foundation

Financial support from the Jenny and Antti Wihuri Foundation is gratefully acknowledged.

### NYU multi-view multi-task model

Nan Wu, Jungkyu Park, and Krzysztof J. Geras are acknowledged for their help in deploying the New York University pre-trained models.

### Research group

The author would like to acknowledge the scientific discussions with Pieta Ipatti, MD, Topi Turunen, MD, Jarmo Reponen, MD, PhD, Satu I. Inkinen, PhD, and Miika T. Nieminen, PhD.

## How to cite

If you found our training pipeline useful, consider citing the repository or the following publications alongside the Wu et al. (2020) publication (cf. References):

> Antti Isosalo, Satu I. Inkinen, Topi Turunen, Pieta S. Ipatti, Jarmo Reponen, & Miika T. Nieminen, "Independent evaluation of a multi-view multi-task convolutional neural network breast cancer classification model using Finnish mammography screening data," Computers in Biology and Medicine 161, 107023 (2023); doi: https://doi.org/10.1016/j.compbiomed.2023.107023

> Antti Isosalo, Satu I. Inkinen, Helinä Heino, Topi Turunen, & Miika T. Nieminen, "MammogramAnnotationTool: Markup tool for breast tissue abnormality annotation," Software Impacts 19, 100599 (2024); doi: https://doi.org/10.1016/j.simpa.2023.100599

## References

* Nan Wu, Jason Phang, Jungkyu Park et al., "Deep neural networks improve radiologists’ performance in breast cancer screening," IEEE Transactions on Medical Imaging, 39(4), 1184-1194 (2020); doi: https://doi.org/10.1109/TMI.2019.2945514
