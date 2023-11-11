# Neural-Cellular-Automata

This project involves the creation of a cellular automata model trained using neural networks, which will be hosted online.

## To Do List

### General

- [x] Project setup
- [x] modularise shaders
- [ ] modularise .js code

### Model Development

- [ ] Life-like CA
  - [ ] Implement a couple different basic rulesets
  - [ ] Dynamic shaders respond to rulestrings

### Model Training

- Waiting for:
  - [ ] continuity
  - [ ] convolutions
- [ ] Create a training method that allows us to find interesting behaviours on the simple implementation as seen [here](https://neuralpatterns.io)

## Resources

This project is currently using the Notion platform to document project progress and important information. This Notion workspace will be made public at a later point in time.

However, some useful resources for this project include:

- Understanding Cellular Automata (CA)
  - Introduction to [Conway's Game of Life](https://playgameoflife.com/)
  - [Explaining CA](https://natureofcode.com/book/chapter-7-cellular-automata/)
  - What are ["Life-Like" CAs](https://en.m.wikipedia.org/wiki/Life-like_cellular_automaton#cite_note-23)
- Implementing CAs
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
  - [WebGPU Shader Tips](https://toji.dev/webgpu-best-practices/dynamic-shader-construction.html)

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

When developing a new feature, create a branch from Dev.

## Dev Cycles

This project will be done in "dev cycles".
That's just an outline how we will conduct development and keep everything organised.
The general progression of a dev cycle is as follows.

1. Branch a new dev branch from main.
2. Create feature branches from dev, and add your new features.
3. Upon completion, feature branches merge back to dev.
4. Bug testing on dev. They may occur as a result of feature integration
5. After bug testing (and more benchmarking maybe) is complete, Then Dev should be merged back to main
6. Dev is archived, and the Dev cycle is complete
