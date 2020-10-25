# Connectionist Temporal Classification (CTC) and its Origin

* In order to manage weakly/unsegmented sequence learning tasks, Connectionist Temporal Classification (CTC) is an output Layer for compatible machine learning models, commonly recurrent neural networks (RNNs), such as LSTM. 

* It is introduced by Alex Graves, Santiago Fernández, Faustino Gomez and Jürgen Schmidhuber, and published in ICML 2006 with a tittle of ***Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks***

## Description

* Unfortunately, the original project was used to be working under Linux OS with the LSTM network and needs several dependencies.

* This code is a refactored version of some part of the original code of the RNNLIB project by Alex Graves.

* In this repository, file [ctc.hpp](./ctc.hpp) is a modified version of the “TranscriptionLayer.hpp” (CTC Layer) file in the original project which does not require any internal or external (ex. Boost C++ Libraries) dependencies, so you can easily include and use it with any standard C++ compiler under any project on  Windows or Linux.

## Changes

* Boost data structure replace with standard C++ STL library.
* There is an option that allows you to easily modify the code to enable/disable the magical blank state!! for more information about this feature read an interesting article by [Théodore Bluche](http://www.tbluche.com/) referenced in the Acknowledgments.

### Dependencies

* Standard C++ 11+
* OS: Windows/Linux/Mac

## Acknowledgments
Inspiration, references, etc.
* [The intriguing blank label in CTC](http://www.tbluche.com/ctc_and_blank.html) 
* [Original library](https://sourceforge.net/projects/rnnl/)
* [CTC paper - ICML 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
