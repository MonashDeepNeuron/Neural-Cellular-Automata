# Descriptions for the Cellular Automata

## What are Cellular Automata?

"Automata" is the plural or automaton, which is roughly a synonym for computer or machine.
Cells are just small things.
Cellular automata then, can be thought of as a bunch of little machines, which follow a rule set, and work together.
Importantly, each cell has *exactly* the same rule set.
Even so, 2 cells may behave very differently at one point in time.
This is similar to how biological cells work.
A bone and a skin cell from the same animal have the same DNA, but have very different properties.

So now, imagine a grid of cells.
Each cell know its own state, as well as the state of its neighbours.
Then, all cells are updated at the same time, and cells may change to a new state.
The change is determined by the cells current state, it's neighbours current state, and the rule set.
In mathematical language, the rule set is a function that takes in a cell and its neighbourhood as inputs, and outputs a new state.

The following classes of cellular automata we showcase all follow this general structure.
However, they have very different definitions of "state", "neighbour" and "rule set".
As such, each of the different classes are capable of a variety of different behaviour.

## Life

John Conway's game of life is likely the most well know cellular automaton.
Each cell is in one of two states; alive or dead.
The instructions are also very simple

- Birth Condition: if dead cell is has exactly 3 alive neighbours, it becomes alive.
- Survival Condition: if a live cell has either 2 or 3 neighbours, it continues to live.
- In all other conditions, a cell dies.

Life is **deterministic**, there is no chance involved.
Yet, life is also **chaotic** and unpredictable.
This is not a contradiction.
A given starting configuration always results in exactly the same behaviours, but even the smallest change to said starting condition may cause Life to behave very differently.
Furthermore, life is Turing complete.
Any computation you can write in any programming language can be converted into some kind of starting position in Life, any when run life, it performs the computation correctly.

One can see that although each cell exhibits simple behaviour, the **emergent** behaviour of some cellular automata is incredibly complex.

We have curated some template starting configurations which showcase the variety of different behaviours of Life.
we have included cyclic configurations (patterns that loop), bunnies (simple patterns that explode in size), and also a simple binary calculator.

## Life like

In life, we had a certain requirements for birth and survival.

moore neighbourhood
rulestrings
so we can change the initial config and also the

## Larger than Life

wacky neighbourhoods
at this scale we don't really observe individual cells do we, more as blobs and stuff.

## Convolving Cellular Automata

differentiable therefore trainable.
whats a convolution

## Vector based Cellular Automata

the state of each cell coresponds to 4 floating point numbers.

## Invisible State Cellular Automata

if we use a vec16, then not all the information is contained in the image file.
so, some of the information is "hidden" from our naked eye.
