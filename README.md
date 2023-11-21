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
- [MonitorBDD](https://github.com/ahmadhatahet/runtime-monitoring/blob/d2503f1881504a6a23786cab1e491a50eebfe0f3/utilities/MonitorBDD.py#L11): The class used to construct the monitor and run the evaluation.
- [Train Model](https://github.com/ahmadhatahet/runtime-monitoring/blob/master/experiments/train-nn-model.ipynb): Notebook to train a NN model.
- [Build a Monitor](https://github.com/ahmadhatahet/runtime-monitoring/blob/master/experiments/build-one-monitor.ipynb): Build a monitor after training a NN model.
- [Analyize NN Stats](https://github.com/ahmadhatahet/runtime-monitoring/blob/master/experiments/analyze-model-stats.ipynb)
- [Analyize a Monitor Stats](https://github.com/ahmadhatahet/runtime-monitoring/blob/master/experiments/monitor-stats.ipynb)
- [Generate Evaluation Data](https://github.com/ahmadhatahet/runtime-monitoring/blob/master/experiments/generate-evaluation-data.ipynb): Generating images used in evaluating the monitor against novel classes.
- [Test Noisy Images](https://github.com/ahmadhatahet/runtime-monitoring/blob/master/experiments/test-noisy-images.ipynb): Add noise gradually to images and test how far the moniotr can identify the images.
- [Test Unseen Class](https://github.com/ahmadhatahet/runtime-monitoring/blob/master/experiments/test-monitor-some-classes.ipynb): Train NN on some classes and leave out a couple of them for testing unseen classes.
- [EDA LHL](https://github.com/ahmadhatahet/runtime-monitoring/blob/master/experiments/eda-lhl.ipynb)

<br />
