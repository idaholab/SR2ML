# SR2ML: Safety Risk and Reliability Model Library

SR2ML is a software package which contains a set of safety and reliability models 
designed to be interfaced with the INL developed RAVEN code. These models can be 
employed to perform both static and dynamic system risk analysis and determine risk 
importance of specific elements of the considered system. Two classes of reliability 
models have been developed; the first class includes all classical reliability models 
(Fault-Trees, Event-Trees, Markov models and Reliability Block Diagrams) which have 
been extended to deal not only with Boolean logic values but also time dependent 
values. The second class includes several components aging models. Models of these 
two classes are designed to be included in a RAVEN ensemble model to perform time 
dependent system reliability analysis (dynamic analysis). Similarly, these models 
can be interfaced with system analysis codes to determine failure time of systems 
and evaluate accident progression (static analysis).


## Available Safety Risk and Reliability Models
- Event Tree (ET) Model
- Fault Tree (FT) Model
- Markov Model
- Reliability Block Diagram (RBD) Model
- Data Classifier
- Event Tree Data Importer
- Fault Tree Data Importer
- Reliability models with time dependent failure rates

## Installation and How to Use?

Please check: https://github.com/idaholab/raven/wiki/Plugins
