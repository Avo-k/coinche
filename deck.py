import random
from tqdm import trange
from itertools import cycle


class Card:
    def __init__(self, value):
        self.value = value % 8
        self.suit = value // 8
        self.isTrump = False
        colors = ["♠", "♥", "♦", "♣"]
        values = ["7", "8", "9", "J", "Q", "K", "10", "A"]
        self._repr = colors[self.suit] + values[self.value]
        self.score = [0, 0, 0, 2, 3, 4, 10, 11][self.value]

    def set_trump(self):
        self.isTrump = True
        value_changes = [0, 1, 6, 7, 2, 3, 4, 5]
        self.value = value_changes[self.value]
        self.score = [0, 0, 3, 4, 10, 11, 14, 20][self.value]
        self.value += 100

    def __repr__(self):
        return self._repr

    def __gt__(self, other):
        return self.value > other.value

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self._repr == other._repr


class Deck:
    def __init__(self):
        self.cards = [Card(i) for i in range(32)]

    def deal(self) -> list:
        assert len(self.cards) == 32
        shuffled = random.sample(self.cards.copy(), 32)
        return [shuffled[i : i + 8] for i in range(0, 32, 8)]

    def remove_trick(self, trick):
        for card in trick:
            self.cards.remove(card)

    def __iter__(self):
        return iter(self.cards)


class Bid:
    def __init__(self, value=None, suit=None):
        self.value = value
        self.suit = suit

    def __repr__(self):
        if self.value:
            return f"{self.value} {['♠', '♥', '♦', '♣'][self.suit]}"
        else:
            return "Pass"

    def __eq__(self, other):
        return self.value == other.value


class Player:
    def __init__(self):
        self.hand = None
        self.trump = None

    def sort_hand(self):
        self.hand.sort(key=lambda x: -x.value + x.suit * 10)

    def init_trump(self, trump: int):
        self.trump = trump
        for card in self.hand:
            if card.suit == trump:
                card.set_trump()
        self.sort_hand()

    def partner_is_leading(self, current_trick, leading_suit):
        if not current_trick:
            return True
        elif len(current_trick) == 1:
            return False
        else:
            updated_values = [
                v.value if v.suit != leading_suit else v.value + 10
                for v in current_trick
            ]
            return max(updated_values) == updated_values[-2]  # partenaire est maitre

    def get_legal_trumps(self, current_trick):
        trumps = [card for card in self.hand if card.isTrump]

        if not trumps:
            return []

        if max(trumps) < max(current_trick):
            return trumps
        else:
            return [card for card in trumps if card > max(current_trick)]

    def get_legal_cards(self, current_trick: list):
        if not current_trick:
            return self.hand

        else:
            leading_suit = current_trick[0].suit
            legal_trumps = self.get_legal_trumps(current_trick)
            if any(
                card.suit == leading_suit for card in self.hand
            ):  # tu as la couleur demandée
                if leading_suit == self.trump:  # il faut monter si c'est l'atout
                    if legal_trumps:
                        return legal_trumps
                    else:
                        return self.hand
                else:
                    return [card for card in self.hand if card.suit == leading_suit]

            if len(current_trick) > 1 and self.partner_is_leading(
                current_trick, leading_suit
            ):  # tu peux pisser
                return self.hand

            if legal_trumps:  # tu coupes si possible
                return legal_trumps

            return self.hand

    def bid(self, passed_bid):
        """given passed bids return a bid"""
        raise NotImplementedError

    def play(self, passed_tricks, current_trick):
        """given a game state return a card"""
        raise NotImplementedError


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
            winner_player = ordered_players[game.wins_trick(current_trick)][0]
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

    def wins_trick(self, trick):
        assert len(trick) == 4
        updated_values = [
            v.value if v.suit != trick[0].suit else v.value + 10 for v in trick
        ]
        return max(range(len(updated_values)), key=updated_values.__getitem__)


class RandomPlayer(Player):
    def __init__(self):
        super().__init__()

    def bid(self, passed_bid):
        """given passed bids return a bid"""
        passed_bid = [bid for bid in passed_bid if bid.value]
        if not passed_bid:  # si pas d'anonce, anonce 80
            return Bid(random.choice([80]), random.randint(0, 3))

        last_bid = passed_bid[-1]
        if last_bid.value < 100 and random.randint(0, 1):  # 50% chance to bid + 10
            return Bid(last_bid.value + 10, random.randint(0, 3))
        else:  # 50% chance to pass
            return Bid()

    def play(self, passed_tricks, current_trick):
        legal_cards = self.get_legal_cards(current_trick)
        chosen_card = random.choice(legal_cards)
        self.hand.remove(chosen_card)
        return chosen_card


class HumanPlayer(Player):
    def __init__(self):
        super().__init__()

    def play(self, passed_tricks, current_trick):
        legal_cards = self.get_legal_cards(current_trick)
        chosen_card = None

        print(f"{current_trick = }")

        while chosen_card not in legal_cards:
            chosen_index = input(
                f"\nYour hand: {self.hand}\n"
                f"Legal cards: {legal_cards} (type 1 for the 1st, 2 for the 2nd ...): "
            )
            chosen_card = legal_cards[int(chosen_index) - 1]

        print(f"You played {chosen_card}")

        self.hand.remove(chosen_card)
        return chosen_card


class BaselinePlayer(Player):
    def __init__(self, cocky=36):
        super().__init__()
        self.cocky = cocky

    @staticmethod
    def card_potential_trump_score(card):
        potential_values = [0, 0, 14, 20, 3, 4, 10, 11]
        return potential_values[card.value]

    def suit_potential_trump_score(self, suit):
        return sum(
            (self.card_potential_trump_score(card) if card.suit == suit else card.score)
            for card in self.hand
        )

    def potential_trump_score(self):
        "return 4 potential trump score, one for each suit"
        return [self.suit_potential_trump_score(suit) for suit in range(4)]

    def get_strongest_suit(self):
        "based on potential trump scores, in case of tie, return suit with the most cards"
        potential_trump_scores = self.potential_trump_score()
        max_score = max(potential_trump_scores)
        candidate_suits = [
            i for i, score in enumerate(potential_trump_scores) if score == max_score
        ]

        # If there's no tie, return the suit with the highest score
        if len(candidate_suits) == 1:
            return candidate_suits[0], max_score

        # In case of a tie, find the suit with the most cards
        suit_card_counts = {
            suit: sum(1 for card in self.hand if card.suit == suit)
            for suit in candidate_suits
        }
        max_cards = max(suit_card_counts.values())

        # Find all suits with the maximum number of cards (in case there's still a tie)
        candidate_suits_with_max_cards = [
            suit for suit, count in suit_card_counts.items() if count == max_cards
        ]
        return candidate_suits_with_max_cards[0], max_score

    def bid(self, passed_bid):
        """given passed bids return a bid"""
        passed_bid = [bid for bid in passed_bid if bid.value]

        strongest_suit, potential_score = self.get_strongest_suit()

        # If no one has bid yet, bid 80 in the strongest suit
        if not passed_bid:
            return Bid(80, strongest_suit)
        elif passed_bid[-1].value < potential_score + self.cocky:
            return Bid(passed_bid[-1].value + 10, strongest_suit)
        else:
            return Bid()

    def is_master(self, remaning_cards, card):
        same_suit = [c.value for c in remaning_cards if c.suit == card.suit]
        return not same_suit or card.value == max(same_suit)

    def choose_best_card(self, passed_tricks, current_trick):
        legal_cards = self.get_legal_cards(current_trick)
        remaining_cards = [
            card for card in Deck() if card not in sum(passed_tricks, [])
        ]
        legal_masters = [
            card for card in legal_cards if self.is_master(remaining_cards, card)
        ]

        if legal_masters:
            return max(legal_masters, key=lambda x: x.value)

        if len(current_trick) < 2:
            return min(legal_cards, key=lambda x: x.value)

        leading_partner = self.partner_is_leading(current_trick, current_trick[0].suit)
        if leading_partner:
            return max(legal_cards, key=lambda x: x.value)
        else:
            return min(legal_cards, key=lambda x: x.value)

    def play(self, passed_tricks, current_trick):
        chosen_card = self.choose_best_card(passed_tricks, current_trick)
        self.hand.remove(chosen_card)
        return chosen_card


if __name__ == "__main__":
    cocky = 36

    binary_scores = [0, 0]
    mega_scores = [0, 0]
    games_won = [0, 0]
    teams = ["Ivan Eloi", "Jean Jules"]

    sample = 10000
    for i in range(sample):
        players = [
            RandomPlayer(),
            BaselinePlayer(cocky=cocky),
            RandomPlayer(),
            BaselinePlayer(cocky=cocky),
        ]
        game = CoincheGame(players, verbose=False)
        scores = game.play()
        mega_scores = [mega_scores[i] + scores[i] for i in range(2)]
        binary_scores = [binary_scores[i] + (scores[i] > 0) for i in range(2)]

        if max(mega_scores) >= 500:
            games_won[mega_scores.index(max(mega_scores))] += 1
            mega_scores = [0, 0]

    print(f"{games_won = }")
    print(
        f"Team {teams[binary_scores.index(max(binary_scores))]} won {(max(binary_scores)/sample)*100:.1f} % of the {sample:_} games."
    )
