# Neural Estimation by Sequential Testing (NEST)
Welcome to the Neural Estimation by Sequential Testing (NEST) Github repository. NEST is a method for simultaneously estimating the psychometric function and generating new trials for a (psychovisual) experiment. The method is currently under peer review. The submitted manuscript can be accessed via [[1]](#1).

In this repository, the code of the NEST method is available. For prospective users of the method, the NEST class encapsulates the entire algorithm such that the method can be used with only a handful of concrete interactions. The user only needs to start up the 
experiment, and then the user can request new trials and submit responses in a loop until the end of the experiment. The script `NEST_example_script.py` contains a simple synthetic two-dimensional function estimation problem that showcases all of the basic communication required to use the method. The pre-evaluated results are already available in the repository in the Results folder. The script `NEST_example_data_analysis.py` shows how the data can be analysed at the end of an experiment, in this case by visualising the estimated function from the previous script. Lastly, the `spatiotemporal_experiment.py` script contains the code for the psychovisual experiment from [[2]](#2) that was performed for this research. 

The code was created using Python 3.8, and the required libraries are given in the requirements.txt file. 

## References
<a id="1">[1]</a> 
Bruin, Sjoerd et al. (2024). 
NEST: Neural Estimation by Sequential Testing. 
ArXiv: https://doi.org/10.48550/arXiv.2405.04226

<a id="2">[2]</a>
Tursun, Cara & Didyk, Piotr (2023).
Perceptual Visibility Model for Temporal Contrast Changes in Periphery.
ACM Transactions on Graphics, Volume 42, Issue 2.
