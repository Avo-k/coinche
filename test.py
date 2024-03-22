import cProfile
import io
import math
import pstats
import random
import copy
from functools import lru_cache
from itertools import chain
from tqdm import trange

BELOTE_REBELOTE = {t: (5 + 8 * t, 6 + 8 * t) for t in range(4)}
SUITS = "♠♥♦♣"


def get_empty_info_dict():
    # __info[idx_player][idx_color] = True if player has no more cards of that color
    return {
        "ruff": [
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ]
    }


def pprint_trick(trick):
    return " ".join(get_card_name(card) for card in trick)


def pprint_tricks(tricks):
    return [pprint_trick(t) for t in tricks]


def bidding_phase(players, current_lead, hands, verbose):

    bids = []
    best_bid = 1

    # while list(map(lambda x: x[1], bids[-3:])) != [None]*3:
    while len(bids) < 4:
        player = (current_lead + len(bids)) % 4
        trump, value = players[player].bid(hands[player], bids)

        assert (
            not value or value > best_bid
        ), f"{players[player].name} tried to bid {value} while best bid is {best_bid}"
        best_bid = value or best_bid

        bids.append((player, value, trump))

        if verbose:
            if value:
                print(f"{players[player].name} bids {value} {SUITS[trump]}")
            else:
                print(f"{players[player].name} passes")

    winning_bid = bids[-4]
    _, value, trump = winning_bid

    for player in range(4):
        if BELOTE_REBELOTE[trump][0] in hands[player] and BELOTE_REBELOTE[trump][1] in hands[player]:
            belote_rebelote = player % 2
            break
    else:
        belote_rebelote = None

    return bids[:-3], belote_rebelote


class GameState:
    __slots__ = [
        "names",
        "bids",
        "bet_value",
        "betting_team",
        "trump",
        "coinche",
        "hands",
        "tricks",
        "leads",
        "current_trick",
        "current_lead",
        "belote_rebelote",
        "last_card_played",
        "scores",
        "info",
    ]

    def __init__(
        self,
        names: list[str],  # 4
        bids: list[tuple[int]],  # [(player, value, trump) ...]
        bet_value: int,  # 80 to 180
        betting_team: int,  # 0: 1st and 3rd players, 1: 2nd and 4th players
        trump: int,  # 0: spades, 1: hearts, 2: diamonds, 3: clubs
        coinche: int,  # 1: no coinche, 2: coinche, 3: surcoinche
        hands: list[list[int]],  # 4 lists of 0 to 8 cards
        tricks: list[list[int]],  # 0 to 8 lists of 4 cards
        leads: list[int],  # 0 to 3 player indices
        current_trick: list[int],  # 0 to 4 cards
        current_lead: int,  # player index, 0 to 3
        belote_rebelote,  # None or team index
        last_card_played: int,  # 0 to 31, -1 for no card
        scores: list[int],  # both scores
        info: dict,  # information about players that everyone knows
    ):
        # assert len(names) == 4, "There must be 4 players"
        # assert not bet_value % 10, "Bet value must be a multiple of 10"
        # assert coinche in range(1, 4), "Coinche must be 1, 2 or 3"
        # assert trump in range(4), "Trump must be 0, 1, 2 or 3"
        # assert len(hands) == 4, "There must be 4 hands"
        # assert 32 == sum(len(hand) for hand in hands) + sum(4 for trick in tricks) + len(
        #     current_trick
        # ), "There must be 32 cards in total"
        # assert all(len(hand) <= 8 for hand in hands), "Each hand must have 8 cards or less"
        # assert len(tricks) <= 8, "There must be 8 tricks at most"

        self.names = names
        self.bids = bids
        self.bet_value = bet_value
        self.betting_team = betting_team
        self.trump = trump
        self.coinche = coinche
        self.hands = hands
        self.tricks = tricks
        self.leads = leads
        self.current_trick = current_trick
        self.current_lead = current_lead
        self.belote_rebelote = belote_rebelote
        self.last_card_played = last_card_played
        self.scores = scores
        self.info = info

    @classmethod
    def fresh_game(
        cls,
        players: list[str] = ["North", "East", "South", "West"],
        current_lead: int = 0,
        seed=None,
    ):

        assert len(players) == 4, "There must be 4 players"

        if seed is not None:
            random.seed(seed)

        deck = list(range(32))
        random.shuffle(deck)

        hands = [deck[i * 8 : i * 8 + 8] for i in range(4)]
        bids, belote_rebelote = bidding_phase(players, current_lead, hands, verbose=False)
        betting_team, bet_value, trump = bids[-1]
        betting_team = betting_team % 2

        for i, hand in enumerate(hands):
            if BELOTE_REBELOTE[trump][0] in hand and BELOTE_REBELOTE[trump][1] in hand:
                belote_rebelote = i % 2
                break
        else:
            belote_rebelote = None

        return cls(
            names=[player.name for player in players],
            bids=bids,
            bet_value=bet_value,
            betting_team=betting_team,
            trump=trump,
            coinche=1,  # TODO: implement coinche in bidding phase
            hands=hands,
            tricks=[],
            leads=[],
            current_trick=[],
            current_lead=current_lead,
            belote_rebelote=belote_rebelote,
            last_card_played=-1,
            scores=[0, 0],
            info=get_empty_info_dict(),
        )

    def copy(self):
        # Perform a deep copy of the game state. Adjust according to your needs.
        return GameState(
            names=self.names.copy(),
            bids=self.bids.copy(),
            bet_value=self.bet_value,
            betting_team=self.betting_team,
            trump=self.trump,
            coinche=self.coinche,
            hands=[hand.copy() for hand in self.hands],
            tricks=[trick.copy() for trick in self.tricks],
            leads=self.leads.copy(),
            current_trick=self.current_trick.copy(),
            current_lead=self.current_lead,
            belote_rebelote=self.belote_rebelote,
            last_card_played=self.last_card_played,
            scores=self.scores.copy(),
            info=copy.deepcopy(self.info),
        )

    def gather_informations(self):
        # update self.info with available informations

        for trick_idx, trick in enumerate(self.tricks):
            leading_suit = trick[self.leads[trick_idx]] // 8

            for player_idx, card in enumerate(trick):
                self.info["ruff"][player_idx][leading_suit] = card // 8 != leading_suit

            # ranks = get_ranks(self.trump, leading_suit)
            # trick_ranked = [ranks[card] for card in trick]
            # relative_player_idx = [(self.leads[trick_idx] + i) % 4 for i in range(4)]

            # self.info["ruff"][relative_player_idx[1]][leading_suit] = trick[relative_player_idx[1]] // 8 != leading_suit

            # partner_is_winning = trick_ranked[0] > trick_ranked[1]
            # if not partner_is_winning:
            #     self.info["ruff"][relative_player_idx[2]][leading_suit] = (
            #         trick[relative_player_idx[2]] // 8 != leading_suit
            #     )

    def get_unseen_cards(self, player_index: int):
        # return list(set(range(32)).difference(*self.tricks, self.hands[player_index], self.current_trick))
        seen_card = self.hands[player_index] + list(chain.from_iterable(self.tricks)) + self.current_trick
        return [card for card in range(32) if card not in seen_card]

    def determinize(self, player_index: int, unseen_cards: list[int]):
        "Return a potential version of the game state where all information that a given player cannot know is randomized"
        if unseen_cards:
            # SUR
            # si un joueur coupe ou pisse, il n'a plus la couleur
            # si un joueur pisse et que partenaire pas maitre, il n'a pas d'atout
            # si un joueur ne monte pas à l'atout alors qu'il devrait, il n'a pas de plus gros atout que le précédent

            # PROBABLE
            # si un joueur a dit belote, il a l'autre carte de la paire roi dame
            # si le joueur annonce une couleur il a au moins 1 carte de cette couleur

            random_hands = [[] for _ in range(4)]
            random_hands[player_index] = self.hands[player_index].copy()

            # unseen_cards_by_suit = [[], [], [], []]
            # for card in unseen_cards:
            #     suit = card // 8
            #     unseen_cards_by_suit[suit].append(card)

            players = list(range(4))
            players.remove(player_index)

            def not_ruff(player, card):
                return not self.info["ruff"][player][card // 8]

            sampling_order = {p: sum(1 for c in unseen_cards if not_ruff(p, c)) for p in players}

            for player in sorted(players, key=sampling_order.get):
                if self.hands[player]:
                    could_be_their_card = [c for c in unseen_cards if not_ruff(player, c)]

                    if len(could_be_their_card) >= len(self.hands[player]):
                        print(f"{pprint_trick(could_be_their_card)} // {pprint_trick(self.hands[player])}")
                        print(pprint_trick(unseen_cards))
                        print(self)
                        quit()

                    assert len(could_be_their_card) >= len(
                        self.hands[player]
                    ), f"{pprint_trick(could_be_their_card)} // {pprint_trick(self.hands[player])}"
                    random_hands[player] = random.sample(could_be_their_card, len(self.hands[player]))
                    unseen_cards = [card for card in unseen_cards if card not in random_hands[player]]

        else:
            random_hands = [hand.copy() for hand in self.hands]

        return GameState(
            names=self.names.copy(),
            bids=self.bids.copy(),
            bet_value=self.bet_value,
            betting_team=self.betting_team,
            trump=self.trump,
            coinche=self.coinche,
            hands=random_hands,
            tricks=self.tricks.copy(),
            leads=self.leads.copy(),
            current_trick=self.current_trick.copy(),
            current_lead=self.current_lead,
            belote_rebelote=self.belote_rebelote,
            last_card_played=self.last_card_played,
            scores=self.scores.copy(),
            info=copy.deepcopy(self.info),
        )

    def __repr__(self):
        betting_team = f"{self.names[self.betting_team]}/ {self.names[self.betting_team+2%4]}"
        contract = f'team {betting_team} joue à {self.bet_value}{["♠", "♥", "♦", "♣"][self.trump]}'
        # ranks = get_ranks(self.trump, self.current_lead)
        # hands = [
        #     f"{name}: {pprint_trick([*sorted(hand, key=lambda x: ranks[x])])}"
        #     for name, hand in zip(self.names, self.hands)
        # ]
        hands = [f"{name}: {pprint_trick([*sorted(hand)])}" for name, hand in zip(self.names, self.hands)]
        passed_tricks = "\n".join(
            [f"{' '.join(get_card_name(card) for card in trick)}" for trick in self.tricks]
            + [" ".join(get_card_name(card) for card in self.current_trick)]
        )
        return "\n---\n".join((contract, "\n".join(hands), passed_tricks)) + "\n---"


def get_higher_trumps(current_trick, hand_trumps, ranks):
    max_trick_trump_value = max(ranks[card] for card in current_trick)
    return [card for card in hand_trumps if ranks[card] > max_trick_trump_value]


if __name__ == "__main__":

    main()

    # # PROFILE
    # pr = cProfile.Profile()
    # pr.enable()
    # main()  # Call the main function where your code is executed
    # pr.disable()
    # pr.dump_stats("profile_results.prof")
    # s = io.StringIO()
    # sortby = "cumulative"
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
