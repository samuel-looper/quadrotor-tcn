# Temporal Convolutions for Multi-Step Quadrotor Motion Prediction

This directory represents the codebase for ongoing research on temporal convolutions for multi-step quadrotor motion prediction. 
This includes all code necessary to develop fully-convolutional Temporal Convolutional Networks (TCNs), hybrid models, and 
physics-based numerical simulations for robotic system modeling. It also includes all code used to train and test predictive models
and generate any published results.

This includes the following files:
- data_loader.py:	Generate custom PyTorch datasets for quadrotor multistep motion prediction
- End2EndNet.py:	Build and train End2EndNet for robotic system modeling
- TCNHybrid.py:		Build and train TCN hybrid models for quadrotor modeling
- PhysicsModel.py:	Build and simulate physics-based quadrotor models
- SystemID.py:		Perform system identification for physics-based quadrotor models
- multi_step_eval.py:	Evaluate robotic system predictive models over multiple steps
- single_step_eval.py:	Evaluate robotic system predictive models over single steps
- prediction_sim.py:	Simulate robotic system motion and predictive models over trajectory samples 
- dataset_stats.py:	Calculate dataset statistics
