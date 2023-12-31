# Descriptions for the Cellular Automata

## What are Cellular Automata?

"Automata" is the plural for automaton, which is roughly a synonym for computer or machine.
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

Two cells are considered neighbours if you can get from one cell to the by moving horizontally, vertically, or diagonally in one move.
This is sometimes referred to as a "moore neighbourhood".
It looks like a square centred at the cell, with side length 3.

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

In Life, birth and survival happen when a cell has 2 or 3 live neighbours.
There is nothing special about these numbers specifically.
We could change the rule set to let birth happen when 8 neighbours are alive, or survive when 0 neighbours are alive.
All modifications in this fashion still produce cellular automata that are quiet similar to Life.

This is the idea behind life-like cellular automata.
It is the class of cellular automata which are similar to Life in the following ways

1. cells are either alive or dead
2. rule sets depend only on the number of live neighbours
3. Moore neighbourhood is used

There are many different rule sets that classify as life-like, but all the cells in a grid must use the *same rule set* for the cellular automata to be a valid.
This way, the DNA analogy still holds.
Each cell uses the same rule set, just now we have the option of trying out different rule sets.

To better understand the behaviour of a general cellular automata, we introduce rule-strings.
It is just an ordered list of numbers representing the conditions to survive, then a forward-slash (the $/$ symbol), then another ordered list of numbers representing birth conditions.

So, if we were to notate life with this rule-string, it would be $23/3$, since survival happens with 2 or 3 live neighbours and birth only happens with exactly 3 neighbours.
Another interesting rule set is $/2$, often called "seeds".
In this rule set, survival never happens. Birth only happens when a cell has 2 live neighbors.
Like real seeds, this rule set has the tendency to explode in size.

There are 2<sup>18</sup> different possible life-like cellular automata,
we won't go through every single one.
However, we do encourage you to try out various rule sets into the provided implementation.
Get a feel for how different rule sets generate cellular automata which behave differently.

There are other ways of notating rule strings.
Some systems put letters before the lists, some convert things back and forth between binary and decimal.
They all represent the same concept.
We've opted for a compact rule string system that is quite popular.

## Larger than Life

The previous cellular automata only dealt with moore neighbourhoods.
Therefore, cells always had exactly 8 neighbours, the ones it touched.
Larger than Life cellular automata (LTL CA for short) drop this requirement,
allowing larger neighbourhoods.
The neighbourhood of a cell could be a square around it,
with side length 7.
One could call this a "moore neighbourhood with range 3", but terms don't matter too much.

Besides moore neighbourhood, another common neighbourhood studied is the "von Neumann" neighbourhood.
A von Neumann neighbourhood of range $n$ consists of all the cells you can get to in $n$ moves, with only horizontal and vertical moves.
It looks like a diamond centred at the cell.
Other neighbourhood shapes are also allowed.

large neighbourhoods mean cells may be affected by other cells far away.
The result is patterns which are observable on a much larger scale.
Individual cells begin to blur together,
and the large-scale behaviour of grid becomes more salient.

Gnarl, bugs, Professor Kellie Evans in her PhD.

## Convolving Cellular Automata

Haven't decided what to call it.

- Convolution Based Cellular Automata
- Convolving Cellular Automata
- Continuous State Cellular Automata

differentiable therefore trainable.
whats a convolution

## Vector based Cellular Automata

the state of each cell corresponds to 4 floating point numbers.

## Invisible State Cellular Automata

if we use a vec16, then not all the information is contained in the image file.
so, some of the information is "hidden" from our naked eye.
