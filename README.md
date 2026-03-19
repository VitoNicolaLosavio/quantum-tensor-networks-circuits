This repository provides an experimental framework for image 
classification based on Variational Quantum Algorithms (VQA). 
The goal is to investigate and compare different quantum 
data encoding strategies and parameterized circuit architectures 
within a hybrid classical–quantum learning setting.

The project is designed in a modular way, allowing flexible integration
of different components of the learning pipeline, including 
feature extraction, quantum encoding, and variational ansatz.
Features are first extracted using classical models and then 
embedded into parameterized quantum circuits, leveraging the 
synergy between classical machine learning 
and quantum machine learning.

Several architectures are implemented, including both end-to-end 
models (with trainable encoding) and non end-to-end models 
(with fixed feature maps), enabling analysis of the role of encoding 
in the learning process. The framework supports multiple encoding 
strategies, such as angle encoding, partial re-uploading, 
and standard re-uploading, designed to enhance expressibility 
while remaining suitable for NISQ devices.

The variational circuits are structured into blocks that combine 
parameterized evolution with interaction layers, where information 
is transferred to ancillary qubits through entangling operations. 
Measurements on the ancilla qubits produce the model output, 
effectively separating representation learning from classification.

The framework supports both binary and multiclass classification, 
using One-vs-All approaches as well as native multiclass 
configurations based on multi-qubit measurements. 
Models are evaluated on benchmark datasets using metrics 
such as accuracy and expressibility, the latter measured via 
KL divergence from the Haar distribution.