# Runtime Monitoring
In this repository we share the implimentation of using binary decision diagram [(BDD)](https://en.wikipedia.org/wiki/Binary_decision_diagram) in abstraction-based monitoring system.
<br />
The code and conducted experiments in this repo assisted in the following master thesis Link(TBD).


# Folder Structure
- [experiments](https://github.com/ahmadhatahet/runtime-monitoring/tree/master/experiments) folder is contains all scripts used to conduct and analyze the monitors.
  - Each dataset has a seperate folder, within it the trained neural network (saved-models) and monitors statistics (bdd) are found.
- [models](https://github.com/ahmadhatahet/runtime-monitoring/tree/master/models) folder contains the architecture of the neural network models.
- [configurations](https://github.com/ahmadhatahet/runtime-monitoring/tree/master/configurations) folder contains the neural network setup like optimizer, regularizations, and last hidden layer size.
- [utilities](https://github.com/ahmadhatahet/runtime-monitoring/tree/master/utilities) folder have some helper functions.
- [exploring-packages](https://github.com/ahmadhatahet/runtime-monitoring/tree/master/exploring-packages) folder exploit some helpful notebooks on the behavior of some packages and/or functions.


# Important Scripts and Notebooks (TBD)
- MonitorBDD: is the actual class used to construct the monitor and run the evaluation.
- Train Model: notebook to train a NN model for a dataset.
- train models
- generate evaluation data
- test noisy images
- test augmented images
- test drop class
- EDA Dataset
- EDA LHL
- EDA monitor

<br />
