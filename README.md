# Neural-Cellular-Automata
This project involves the creation of a cellular automata model trained using neural networks, which will be hosted online.

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
 
More resources will be made available in this repo as the project progresses.


## REPO Maintenance 

### Main Branch

This branch should **always** be operational. 
it only branches to the dev branch. Only the dev branch merges to it

pushing straight to main should only be done for 
- Documentation
- md files
- comments i guess

### Dev Branch

fixing bugs. there is only one dev branch. Only merges to the main branch when fully operational and fully tested. When merged, archive the dev branch with the date of archiving. 

### Feature Branches

When developing a new feature, create a branch from Dev. 


### Dev Cycles

1. Branch a new Dev Branch
2. Create Feature branches from Dev, and add your new features. Benchmarking of individual features may be performed here. 
3. Features back to Dev
4. Bug testing and bug fixing to occur here. Bugs are likely a result of multiple new features not working well with each other, so the main thing to test for is that all the new things are playing nicely with each other
5. After bug testing (and more benchmarking maybe) is complete, Then Dev should be merged back to main
6. Dev is archived, and the Dev cycle is complete

