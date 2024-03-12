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
