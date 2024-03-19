from game import SimpleCoincheGame
from players import RandomPlayer

game = SimpleCoincheGame(
    agents=[RandomPlayer(), RandomPlayer(), RandomPlayer(), RandomPlayer()],
    current_lead=0,
    verbose=False,
)

game.deal()
game.bidding()

done = False

while not done:
    done = game.step()

print(game.scores)
