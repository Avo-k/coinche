import random
from tqdm import trange
from itertools import cycle
from utils import Deck, Bid
from players import RandomPlayer, BaselinePlayer, HumanPlayer
import copy


class CoincheGame:

    names = ["Ivan", "Jean", "Eloi", "Jules"]

    def __init__(self, players, lead=random.randint(0, 3), verbose=False):
        self.players = players
        self.verbose = verbose
        self.tricks = []
        self.scores = [0, 0]
        self.lead = lead
        self.current_trick = []

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
            if bid.value:
                assert not bid.value % 10 and all(b.value < bid.value for b in passed_bid if b.value)
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
        updated_values = [v.value if v.suit != trick[0].suit else v.value + 10 for v in trick]
        return max(range(len(updated_values)), key=updated_values.__getitem__)

    def play(self):
        for i in range(8):
            ordered_players = self.get_ordered_players()
            if self.verbose:
                print(*zip(self.names, [p.hand for p in self.players]), sep="\n")
                print(f"\n{self.names[ordered_players[0][0]]} starts the trick.")

            for idx, player in ordered_players:
                card = player.play(self)
                self.current_trick.append(card)
            score = sum(card.score for card in self.current_trick)
            winner_player = ordered_players[self.wins_trick(self.current_trick)][0]
            self.lead = winner_player
            self.scores[winner_player % 2] += score
            if self.verbose:
                print(f"{self.names[winner_player]} wins {self.current_trick = } for {score} points.")
                print("-" * 30)
            self.tricks.append(self.current_trick)
            self.current_trick = []
        self.scores[winner_player % 2] += 10  # 10 de der

        if self.target_score <= self.scores[self.bidding_team]:
            self.scores[self.bidding_team] = self.target_score
            self.scores[self.bidding_team ^ 1] = 0
        else:
            self.scores[self.bidding_team] = 0
            self.scores[self.bidding_team ^ 1] = 160

        return self.scores

    def copy(self):
        "Return a deep copy of the game state."
        newplayers = [copy.deepcopy(player) for player in self.players]
        new_game = CoincheGame(players=newplayers, lead=self.lead, verbose=self.verbose)
        new_game.tricks = self.tricks.copy()
        new_game.scores = self.scores.copy()
        new_game.current_trick = self.current_trick.copy()
        return new_game

    def is_terminal(self):
        return len(self.tricks) == 8


def get_card_name(index):
    values = ["7", "8", "9", "10", "J", "Q", "K", "A"]
    suits = ["♠", "♥", "♦", "♣"]
    return values[index % 8] + suits[index // 8]


CARD_STR = {i: get_card_name(i) for i in range(32)}


class SimpleCoincheGame:

    def __init__(self, current_lead, verbose=False):
        self.verbose = verbose
        self.tricks = []
        self.scores = [0, 0]

        self.current_lead = current_lead
        self.current_trick = []

    def deal(self):
        # Donne
        self.deck = list(range(32))
        random.shuffle(self.deck)
        self.hands = [self.deck[i : i + 8] for i in self.deck]
        self.deck = set(range(32))

    def bidding(self):
        # Annonces
        passed_bid = []  # [(value: int, suit: int)]
        cycle_players = cycle(enumerate(self.agents))
        # until there is 3 consecutive pass
        while len(passed_bid) < 4 or passed_bid[-3:] != [(0, 0)] * 3:
            idx, player = next(cycle_players)
            bid = player.bid(passed_bid)
            if bid[0]:
                assert not bid[0] % 10 and all(b[0] < bid[0] for b in passed_bid if b[0])
            passed_bid.append(bid)
        winner_idx, _ = next(cycle_players)
        winner_bid = passed_bid[-4]

        self.bidding_team = winner_idx % 2
        self.target_score = winner_bid[0]
        self.trump = winner_bid[1]

        non_trump_scores = [0, 0, 0, 10, 2, 3, 4, 11]
        trump_scores = [0, 0, 14, 10, 20, 3, 4, 11]
        self.card_scores = non_trump_scores * self.trump + trump_scores + non_trump_scores * (3 - self.trump)

    def is_terminal(self):
        return len(self.tricks) == 8

    def get_next_agent_index(self):
        return (self.lead + len(self.current_trick)) % 4

    def get_card_value(self, card):
        value, suit = card // 8, card % 8
        if suit == self.trump:
            value += 100
        if suit == self.current_lead:
            value += 10
        return value

    def get_card_score(self, card):
        return self.card_scores[card]

    def trick_results(self, trick):
        "from a trick, return the winner index (in the trick) and points scored."
        values = [self.get_card_value(card) for card in trick]
        winner_index = values.index(max(values))
        score = sum(self.get_card_score(card) for card in trick)
        return winner_index, score

    def step(self):
        agent = self.agents[(self.current_lead + len(self.current_trick)) % 4]
        card = agent.play()
        assert card in self.hands[self.current_lead]
        self.current_trick.append(card)
        if len(self.current_trick) == 4:
            winner_index, score = self.trick_results(self.current_trick)
            self.current_lead = (winner_index + self.current_lead) % 4
            self.scores[self.current_lead % 2] += score
            self.tricks.append(self.current_trick)
            self.current_trick = []
        return self.is_terminal()
