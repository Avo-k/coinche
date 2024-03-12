from utils import Deck, Bid, Player
import random


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

    def play(self, game_state):
        passed_tricks, current_trick = game_state.tricks, game_state.current_trick
        legal_cards = self.get_legal_cards(current_trick)
        chosen_card = random.choice(legal_cards)
        self.hand.remove(chosen_card)
        return chosen_card


class HumanPlayer(Player):
    def __init__(self):
        super().__init__()

    def play(self, game_state):
        passed_tricks, current_trick = game_state.tricks, game_state.current_trick
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

    def play(self, game_state):
        passed_tricks, current_trick = game_state.tricks, game_state.current_trick
        chosen_card = self.choose_best_card(passed_tricks, current_trick)
        self.hand.remove(chosen_card)
        return chosen_card
