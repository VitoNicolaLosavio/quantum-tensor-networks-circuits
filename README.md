## Overview

This repository provides an experimental framework for image classification 
based on Variational Quantum Algorithms (VQAs). 
The main objective is to investigate and compare different 
quantum data encoding strategies and parameterized circuit architectures
within a hybrid classical–quantum learning setting.

## Project Structure

The framework is designed with a modular architecture, enabling flexible 
integration of the main components of the learning pipeline, 
including feature extraction, quantum encoding, and variational ansätze.

Classical models are first used to extract features from images, 
which are then embedded into parameterized quantum circuits. 
This approach leverages the synergy between classical machine learning 
and quantum machine learning.

## Implemented Architectures

Several architectures are available:

- **End-to-end models**, where the encoding is trainable  
- **Non end-to-end models**, where feature maps are fixed  

This design allows for a systematic analysis of the impact of 
encoding strategies on model performance.

The framework supports multiple encoding techniques, including:

- Angle encoding  
- Partial data re-uploading  
- Standard data re-uploading  

These methods aim to improve circuit expressibility while remaining compatible with NISQ devices.

## Variational Circuits with interaction layer

The variational circuits are organized into blocks that combine parameterized evolution with interaction layers. 
Information is transferred to ancillary qubits through entangling operations,
and measurements on these ancilla qubits produce the final model output. 
This structure enables a separation between representation learning and classification.

## Classification Tasks

The framework supports both:

- Binary classification  
- Multiclass classification  

Multiclass tasks can be handled using:

- One-vs-All strategies  
- Native multiclass configurations based on multi-qubit measurements  

## Evaluation

Models are evaluated on benchmark datasets using metrics such as:

- Accuracy  
- Expressibility (measured via KL divergence from the Haar distribution)

## Stroke Dataset Application

The framework includes an application to medical image classification using the **Brain Stroke CT Dataset** from Kaggle:  
https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset

### Setup

To run experiments on this dataset:

1. Download the dataset from Kaggle  
2. Extract the images  
3. Place them inside the `data/` directory of the repository

### MNIST Dataset

The framework also supports experiments on the **MNIST dataset**, 
a standard benchmark for handwritten digit classification.

## Results

All experimental results are available in the `results/` folder.