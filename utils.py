import random


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
