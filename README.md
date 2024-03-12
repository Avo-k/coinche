# Solving belote-coinchée
***a game theory experiment***

Games can be classified according to the following properties:

- **Zero-sum**: Whether the reward to all players sums to zero (or equivalently some constant value) for all terminal states.
- **Information**: Whether the state of the game is fully or partially observable to the players.
- **Determinism**: Whether or not taking the same action from the same state always results in the same outcome.
- **Sequential**: Whether actions are applied by players sequentially or simultaneously.
- **Discrete**: Whether actions are discrete or applied in real-time.

for now I only have solved 2-player, zero-sum, perfect-information, determinist, sequential and discrete games. Like Chess, or utilimate tic-tac-toe. Now is the time to tackle partially observable games, starting with the belote-coinchée because I love this game.

When MCTS or Minimax algorithms were enough for those games ; exploring the future game tree. Now another dimension is to be considered, and lots of different game tree are possible from a single player perspective. 

We then have to consider what information do we have, to determine which future game trees we should explore.

## La coinche

TODO: brief rules explanation

## What currently works

I currently have a working coinche environment and 3 bots:

### Random player 
- **Betting**: toss a coin to bet +10 or pass until 100 is reached
- **Playing**: play random legal move

### Human player 
- **Betting**: human input
- **Playing**: human input

### Baseline player 
- **Betting**: bet on their best suit, how high is based on their number of point in hand + a "cocky" variable, to adjust how daring the bot is
- **Playing**: play if has the best card in a suit, else lowest card


## TODO

### Oracle player

They knows every hand and always play the best move. They consider every player is also an Oracle. I shall use MCTS to make them.


### Skorm player

The AI. Only know what a regular player would. Play their best. ISMCTS ?