# UoM Dissertation project folder

## Description

The project aims to improve the algorithm,DDQL-MI, presented in the paper "rivacy-Cost Management in Smart Meters With Mutual-Information-Based Reinforcement Learning" by introducing new assumptions to the derivation of the per-step privacy signal. This results in a continuous probability distribution learnt by the H-network in the paper, allows a RL algorithm to produce charging/discharging rate in a continuous action space, instead of a discrete action space. A custom implementation of the DDQL-MI following available information in this paper and another paper from the same author is provided as a baseline for comparison.

A custom RL environment mimicking the energy management unit is produced, which includes a rechargeable battery with adjustable efficiency, a H-network to be trained along with.

For info about dataset creation, please go to [README](dataset/README.md)

## Environment Setup

Setup the conda environment using `environment.yml`.