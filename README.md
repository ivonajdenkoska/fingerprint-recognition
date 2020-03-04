# Fingerprint recognition

### This project was part of the assignments for the [Biometrics System Concepts course](https://onderwijsaanbod.kuleuven.be/syllabi/e/H02C7AE.htm#activetab=doelstellingen_idp2852640) @ [KU Leuven](https://www.kuleuven.be/english/), for the academic year 2018/2019. Parts of the code and the algorithms were provided during the course.

Fingerprint recognition is probably the most mature biometric technique, which finds lots of real - life applications since long ago. This project explores fingerprint recognition from two points of view: identification and authentication. 

### Data
The data consists of images of fingerprints, provided as part of [FVC2002](http://bias.csr.unibo.it/fvc2002/databases.asp): the Second International Competition for Fingerprint Verification Algorithms. There are 4 available datasets and we chose to work with the second dataset: DB2, which consists of 80 images of fingerprints that belong to 10 different individuals (or classes), which gives 8 images per person.

### Enhancement techniques
Each fingerprint image is converted to grayscale and it is enchanced by using the [Fingerprint-Enhancement-Python](https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python) library for fingerprints enhancement, cloned in the folder src/fprmodules/enhancement. The library uses oriented Gabor filter (a linear filter in image processing used for texture analysis) to enhance the fingerprint image. After you clone the library in the specified folder location, make sure to use the following return statement: ```return(enhim, mask, orientim, freqim)``` for the image_enhance function.

### Implementation
The implementation is entierly done in Python and the main flow of the project is assembled in a Jupyter notebook fingerprint-recognition.ipynb, located in the src folder.





