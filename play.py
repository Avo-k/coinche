from players import RandomPlayer, BaselinePlayer, HumanPlayer
from game import CoincheGame
from oracle_mcts import OraclePlayer


def play_game(players):
    game = CoincheGame(players, verbose=False)
    scores = game.play()
    return scores


if __name__ == "__main__":

    binary_scores = [0, 0]
    mega_scores = [0, 0]
    games_won = [0, 0]
    teams = ["Ivan Eloi", "Jean Jules"]

    sample = 10
    for i in range(sample):
        players = [
            RandomPlayer(),
            OraclePlayer(),
            RandomPlayer(),
            BaselinePlayer(),
        ]
        scores = play_game(players)
        mega_scores = [mega_scores[i] + scores[i] for i in range(2)]
        binary_scores = [binary_scores[i] + (scores[i] > 0) for i in range(2)]

        if max(mega_scores) >= 500:
            games_won[mega_scores.index(max(mega_scores))] += 1
            mega_scores = [0, 0]

    print(f"{games_won = }")
    print(
        f"Team {teams[binary_scores.index(max(binary_scores))]} won {(max(binary_scores)/sample)*100:.1f} % of the {sample:_} games."
    )
