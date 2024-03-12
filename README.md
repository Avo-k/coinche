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

We then have to consider what information do we have, to guess which future game trees we should explore.

