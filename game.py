import random
from tqdm import trange
from itertools import cycle
from utils import Deck, Bid
from players import RandomPlayer, BaselinePlayer, HumanPlayer


class CoincheGame:

    names = ["Ivan", "Jean", "Eloi", "Jules"]

    def __init__(self, players, lead=random.randint(0, 3), verbose=False):
        self.players = players
        self.verbose = verbose
        self.tricks = []
        self.scores = [0, 0]
        self.lead = lead

        self.deck = Deck()
        hands = self.deck.deal()
        for i, player in enumerate(self.players):
            player.hand = hands[i]

        # Annonces
        self.bidding_team, self.target_score, self.trump = self.bidding()

        # self.trump = random.randint(0, 3)
        for player in self.players:
            player.init_trump(self.trump)

    def bidding(self):
        passed_bid = []
        cycle_players = cycle(self.get_ordered_players())
        # until there is 3 consecutive pass
        while len(passed_bid) < 4 or passed_bid[-3:] != [Bid()] * 3:
            idx, player = next(cycle_players)
            bid = player.bid(passed_bid)
            passed_bid.append(bid)

        winner_idx, _ = next(cycle_players)
        winner_team_idx = winner_idx % 2

        if self.verbose:
            print("Bidding phase :")
            winner_name = self.names[winner_idx]

            cycle_players = cycle(self.get_ordered_players())
            for bid in passed_bid:
                idx, player = next(cycle_players)
                print(end=f"{self.names[idx]}: {bid}, ")

            print(f"\n{winner_name} won the bet with {passed_bid[-4]}")

            print("-" * 30)

        return winner_team_idx, passed_bid[-4].value, passed_bid[-4].suit

    def get_ordered_players(self):
        players = self.players[self.lead :] + self.players[: self.lead]
        indexes = [p % 4 for p in range(self.lead, self.lead + 4)]
        return list(zip(indexes, players))

    def wins_trick(self, trick):
        assert len(trick) == 4
        updated_values = [
            v.value if v.suit != trick[0].suit else v.value + 10 for v in trick
        ]
        return max(range(len(updated_values)), key=updated_values.__getitem__)

    def play(self):
        for i in range(8):
            current_trick = []
            ordered_players = self.get_ordered_players()
            if self.verbose:
                print(*zip(self.names, [p.hand for p in self.players]), sep="\n")
                print(f"\n{self.names[ordered_players[0][0]]} starts the trick.")
            for idx, player in ordered_players:
                card = player.play(self.tricks, current_trick)
                current_trick.append(card)
            score = sum(card.score for card in current_trick)
            winner_player = ordered_players[self.wins_trick(current_trick)][0]
            self.lead = winner_player
            self.scores[winner_player % 2] += score
            if self.verbose:
                print(
                    f"{self.names[winner_player]} wins {current_trick = } for {score} points."
                )
                print("-" * 30)
            self.tricks.append(current_trick)
        self.scores[winner_player % 2] += 10  # 10 de der

        if self.target_score <= self.scores[self.bidding_team]:
            self.scores[self.bidding_team] = self.target_score
            self.scores[self.bidding_team ^ 1] = 0
        else:
            self.scores[self.bidding_team] = 0
            self.scores[self.bidding_team ^ 1] = 160

        return self.scores
