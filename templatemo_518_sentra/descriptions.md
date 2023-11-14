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
- In all other conditions, a cell dies

Life is **deterministic**, there is no chance involved.
Yet, life is also **chaotic** and unpredictable.
This is not a contradiction.
A given starting position always results in exactly the same behaviours, but even the smallest change to said starting condition may cause Life to behave very differently.

We have curated some templates which showcase the variety of different things
what are the simple rules.
Deterministic, chaotic, emergent.
tecnhnically, this is the moore neighbourhood.

## Life like

## Larger than Life

## Convolving Cellular Automata

## vector 4 based Cellular Automata

## vector 16 based Cellular Automata

that we can have hidden information.
