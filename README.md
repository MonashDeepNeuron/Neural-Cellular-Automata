# Neural-Cellular-Automata

## Goals

Our goals can be broken down into answering three topics:

1. What are NCA? How is NCA different from other NCA?
2. What can NCA be used for? Does NCA provide an advantage over other similar architectures?
3. How can NCA be improved?

As a result of answering these questions, we aim to produce a research paper.

## Open Source Website!

All of our code for <a href="https://neuralca.org">our website</a> and for generating website content is open source and available here! There are two main parts:

1. **The Training Code:** This produces the weights an biases of our neural network. All implemented in **PyTorch**. There will be multiple versions of this for various experiments.
2. **The Website:** Models are rendered in **WebGPU** and **JavaScript.** Output is what you see in front of you!

## Current Progress

### Website Construction
- [x] Learning WebGPU and creating a platform to share our work
  - [x] Built Conway's game of life
  - [x] Built Life-like CA, capable of taking input changing the behaviour of the CA
  - [x] Built Larger than Life, capable of creating coloured outputs 
  - [x] Built Continuous, implementing convolutions and live-editable WebGPU code
  - [x] Built Growing Neural Cellular Automata
    - [x] Developed and trained model
    - [x] Ported model weights over to the website in readable format
    - [ ] Implemented Growing Neural Cellular Automata on website
- [ ] Built the about page
- [x] Created basic descriptions of Neural Cellular Automata
- [ ] Created advanced resources on Neural Cellular Automata

### Model Development
- [x] Implement our own copy of Growing Neural Cellular Automata in PyTorch
  - [x] Implement model
  - [x] Implement Growing training script
  - [x] Implement Persisting training script
  - [ ] Implement Regenerating training script
- [x] Create system to weights and convert to format usable for website
- [x] Explore the viability of extending NCA for Image Segmentation
  - [x] Implementation of the paper Image Segmentation Neural Cellular Automata
  - [x] Implementation of the paper Med-NCA
- [x] Explore the viability of extending NCA for texture generation
  - [x] Implementation of Self Organising Textures

## Resources

This project is currently using the Notion platform to document project progress and important information. This Notion workspace will be made public at a later point in time.

However, some useful resources for this project include:
- Understanding Cellular Automata (CA)
  - Introduction to [Conway's Game of Life](https://playgameoflife.com/)
  - [Explaining CA](https://natureofcode.com/book/chapter-7-cellular-automata/)
  - What are ["Life-Like" CAs](https://en.m.wikipedia.org/wiki/Life-like_cellular_automaton#cite_note-23)
- Neural Cellular Automata
  - [Neural Patterns](https://neuralpatterns.io)
  - [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
- Tutorials for Building CAs
  - [Daniel Shiffman's Tutorial](https://www.youtube.com/watch?app=desktop&v=DKGodqDs9sA)
  - [Building Conway's Game of Life using WebGPU](https://codelabs.developers.google.com/your-first-webgpu-app#0)
  - [Physics Simulation with CA](https://www.youtube.com/watch?v=VLZjd_Y1gJ8&pp=ygUfY2VsbHVsYXIgYXV0b21hdGEgc2FuZCBwYXJ0aWNsZQ%3D%3D)
- Other
  - [Noita](https://en.wikipedia.org/wiki/Noita_(video_game)#cite_note-11) uses CA to make physics simulation
  - ["Rule-String" Notation](https://conwaylife.com/wiki/Rulestring)
  - [CAs and Computational Systems](https://direct.mit.edu/isal/proceedings/isal2021/33/105/102949)
  - [Emergent Gardens](https://www.youtube.com/@EmergentGarden)
- WebGPU Resources
  - [WebGPU Shader Tips](https://toji.dev/webgpu-best-practices/dynamic-shader-construction.html)
  - [Typescript & WebGPU Examples](https://webgpu.github.io/webgpu-samples/samples/helloTriangle)

More resources will be made available in this repo as the project progresses.

## Repository Maintenance

We will maintain at most 3 **types** of branches on the repository for this project.
How each branch type will be used is detailed in [dev cycles](#dev-cycles).
Details on each branch type are elaborated here:

### Main Branch

The branch that is **always** operational.
It contains features that have been tested, are fully functional, and are ready to be seen by the public.
As such, this branch is to be treated extremely carefully.
Main will **only** branch to dev, and only dev will merge to main.
The reason for this is so we can vet and check everything before anything goes up to main.
Pushing straight to main should only be done **never**.

### Dev Branch  

This is the branch that hold all things under development. Every time a feature is finished, it will be merged to dev.
On dev we then have the chance to do bug fixing and benchmarking.
We do all the necessary test to ensure the stuff we upload is genuinely ready.

### Feature Branches

These branches are where actual programming occurs.
Create a feature branch from main, and name it accordingly.
Try to keep it down to one feature per brach, so the git history will clearly outline our development process.

There is a lot of freedom in what you are allowed to do in this branch type, since everything will be vetted when you merge to dev, and more vetting will be done before merging to main.
The principle is just try to keep things reasonable and clear.

## Dev Cycles

This project will be done in "dev cycles".
That's just an outline how we will conduct development and keep everything organised.
The general progression of a dev cycle is as follows.

1. Branch a new dev branch from main.
2. Create feature branches from dev, and add your new features.
3. Upon completion, feature branches merge back to dev.
4. Bug testing on dev. These may occur as a result of feature integration.
5. After bug testing (and more benchmarking maybe) is complete, Then Dev should be merged back to main.
6. Dev is archived, and the Dev cycle is complete.

## Setup
This project works on Python 3.12 and may also work on versions newer than Python 3.9.  To install and manage multiple versions of Python, we recommend using [miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install).  This is especially useful, if you plan on using Python for other projects.

### Conda Environment
To create a new environment for this project, run the following commands in your terminal (assuming you have a GPU that supports CUDA 12.4):
```bash
conda create -n nca --file requirements.txt
```

If you do not have a discrete GPU, or have an older GPU that does not support CUDA 12.4, you can create an environment manually with the following commands:
```bash
conda create -n nca python==3.12
conda activate nca
pip install matplotlib
```

You will also need to install the correct PyTorch version with conda using their [installation guide](https://pytorch.org/get-started/locally/).  Ensure that you choose `Conda` as the package manager, and select the correct CUDA version for your GPU or CPU.

### Using the Environment
To use the environment, run the following command in your terminal:
```bash
conda activate nca
```

You can also automatically activate the environment for this project in VSCode by selecting the correct Python interpreter in the bottom right corner of the window (when a Python file is open).
