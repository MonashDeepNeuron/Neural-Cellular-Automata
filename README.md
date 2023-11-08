# Neural-Cellular-Automata

This project involves the creation of a cellular automata model trained using neural networks, which will be hosted online.

## To Do List

### Set up (Complete by 28/11/23)

- [ ] Setup javascript environment, work it with HTML

### High Performance Computing (In Progress)

- [x] WebGPU game of life implementation (Tutorial)
  - [ ] Be able to highlight cells with cursor (optional)
  - [ ] Be able to move forward one frame on button press (for bug testing)
  - [ ] Integrate HTML and javascript
- [ ] JS program that produces a parallel “Lifelike CA”
  - Input: String
  - Output: Functioning CA Shader
  - Understand Lifelike CA notation
- [ ] Implement Continuous CA
  - Rules are now determined by convolutions
- [ ] Implement LTL Continuous CA
- [ ] Implement Vector Cells (16)

### Deep Learning (Pending Until HPC Model Sufficiently Developed)

- Pending model is sufficiently developed:
  - [ ]continuity
  - [ ]convolutions

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
  - [Noita](<https://en.wikipedia.org/wiki/Noita_(video_game)#cite_note-11>) uses CA to make physics simulation
  - ["Rule-String" Notation](https://conwaylife.com/wiki/Rulestring)
  - [CAs and Computational Systems](https://direct.mit.edu/isal/proceedings/isal2021/33/105/102949)
  - [Emergent Gardens](https://www.youtube.com/@EmergentGarden)

More resources will be made available in this repo as the project progresses.

## REPO Maintenance 

### Dev Cycles

A development cycle refers to a period of time over which the team develops the project. 
At the end of each dev cycle, main is updated. 
So, people will be able to see the updates we've made *after* a dev cycle finishes.
Here's what a dev cycle will look like:

1. A new Dev branch is created from main.
2. Feature branches are created from Dev. Features are developed and benchmarked. 
3. Features merge back to Dev when completed. **Don't** merge features that aren't fully operational yet!
4. Bug testing. Features may not work together, so we have check that here.
5. More bug testing, and benchmarking. 
6. when all features are integrated, Dev is merged to main.
7. Dev is archived, and the Dev cycle is complete.

See below for more details on how each branch type should operate. 

### Main Branch

This branch should **always** be operational. 
it only branches to the dev branch. 
Only the dev branch merges to it. 
Its essentially public facing, so it should always look pretty

Pushing straight to main should only be done for documentation, md files, and comments. 
But even then just put it on dev, why does it need to be on main, it'll get onto main the at the end of the dev cycle anyway. 

### Dev Branch

There is at most one dev branch at any point in time.
if the project is on break, then there should be no dev branch.  
Only merges to the main branch when fully operational and fully tested. 
After merged, archive the dev branch with the date of archiving. 

### Feature Branches

These branch from dev. 
Each feature should have its own branch. 
Name branches accordingly. 

Make sure features are fully operational before merging. 
This is so that if there are issues on dev, it's because of feature integration, not because individual features are broken. 

### File Management Guide

I guess stuff here isn't necessary, but highly, highly preferable. The general idea is to keep things as modular as possible. 

1. One file should only have one language written in it. So the tutorial file is **bad** because the Javascript module is written directly in the html file. 
2. One file should only have one method in it.
3. Methods that pertain to a similar feature should be put in a folder together. 