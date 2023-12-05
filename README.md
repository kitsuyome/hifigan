# HiFi-GAN Implementation Repository

## Objective

This repository is based on the paper  'HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis'. The original paper can be found here: [HiFi-GAN Paper](https://arxiv.org/abs/2010.05646).

## Report

[Wandb link](https://wandb.ai/kitsuyomi/dla-hw-4/reports/HiFi-GAN-Implementation--Vmlldzo2MTc2ODAx)

## Installation

Clone the repository and install dependencies:

```
!git clone https://github.com/your-github-username/hifigan.git
!cd hifigan
!pip install -r requirements.txt
```

## Synthesizing

Run the setup script to download data, prepare the environment and checkpoints:

```
!chmod +x setup_inference.sh
!./setup_inference.sh
!python inference.py
```

Not it generates three test audio files, as required for the homework assignment

## Reproduce Training

Run the setup script to download data and prepare the environment:

```
!chmod +x setup_train.sh
!./setup_train.sh
!python train.py
```

## License

[MIT License](LICENSE)
