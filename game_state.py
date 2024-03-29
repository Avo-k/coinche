import cProfile
import io
import math
import pstats
import random
import copy
from functools import lru_cache
from itertools import chain
from tqdm import trange

# BELOTE_REBELOTE = {0: (5, 6), 1: (13, 14), 2: (21, 22), 3: (29, 30)}
BELOTE_REBELOTE = {t: (t * 8 + 5, t * 8 + 6) for t in range(4)}
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
        self.last_card_played = last_card_played
        self.scores = scores
        self.info = info

    def __hash__(self):
        return hash(
            (
                self.bet_value,
                self.betting_team,
                self.trump,
                tuple(tuple(hand) for hand in self.hands),
                tuple(tuple(trick) for trick in self.tricks),
                tuple(self.leads),
                tuple(self.current_trick),
                self.current_lead,
                tuple(self.scores),
            )
        )

    @classmethod
    def fresh_game(
        cls,
        players: list[str] = ["North", "East", "South", "West"],
        hands: list[list[int]] = None,
        current_lead: int = None,  # 0 to 3
        seed=None,
        verbose=False,
    ):

        assert len(players) == 4, "There must be 4 players"

        if seed is not None:
            random.seed(seed)

        if current_lead is None:
            current_lead = random.randint(0, 3)

        if hands is None:
            deck = list(range(32))
            random.shuffle(deck)
            hands = [deck[i * 8 : i * 8 + 8] for i in range(4)]

        if verbose:
            for p, h in zip(players, hands):
                print(f"{p.name}: {pprint_trick(list(sorted(h)))}")

        bids = bidding_phase(
            players,
            current_lead,
            hands.copy(),
            verbose=verbose,
        )
        betting_team, bet_value, trump = bids[-1]
        betting_team = betting_team % 2

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
            last_card_played=-1,
            scores=[0, 0],
            info=get_empty_info_dict(),
        )

    def copy(self, hands=None):
        return GameState(
            names=self.names.copy(),
            bids=self.bids.copy(),
            bet_value=self.bet_value,
            betting_team=self.betting_team,
            trump=self.trump,
            coinche=self.coinche,
            hands=[hand.copy() for hand in self.hands] if hands is None else hands,
            tricks=[trick.copy() for trick in self.tricks],
            leads=self.leads.copy(),
            current_trick=self.current_trick.copy(),
            current_lead=self.current_lead,
            last_card_played=self.last_card_played,
            scores=self.scores.copy(),
            info={"ruff": [p.copy() for p in self.info["ruff"]]},
        )

    def gather_informations(self):
        # update self.info with available informations

        for trick_idx, trick in enumerate(self.tricks):
            leading_suit = trick[self.leads[trick_idx]] // 8

            for player_idx, card in enumerate(trick):
                self.info["ruff"][player_idx][leading_suit] = card // 8 != leading_suit

        if len(self.current_trick) > 1:
            leading_suit = self.current_trick[0] // 8
            for trick_idx, card in enumerate(self.current_trick):
                self.info["ruff"][(self.current_lead + trick_idx) % 4][leading_suit] = card // 8 != leading_suit

        # TODO: add more information

    def get_unseen_cards(self, player_index: int):
        # return list(set(range(32)).difference(*self.tricks, self.hands[player_index], self.current_trick))
        seen_card = self.hands[player_index] + list(chain.from_iterable(self.tricks)) + self.current_trick
        return [card for card in range(32) if card not in seen_card]

    def can_own(self, player, card):
        return not self.info["ruff"][player][card // 8]

    def get_next_illegal_cards(self, hands):
        for p, hand in enumerate(hands):
            for card in hand:
                if not self.can_own(p, card):
                    return p, card
        return None

    # SUR
    # si un joueur coupe ou pisse, il n'a plus la couleur
    # si un joueur pisse et que partenaire pas maitre, il n'a pas d'atout
    # si un joueur ne monte pas à l'atout alors qu'il devrait, il n'a pas de plus gros atout que le précédent

    # PROBABLE
    # si un joueur a dit belote, il a l'autre carte de la paire roi dame
    # si le joueur annonce une couleur il a au moins 1 carte de cette couleur

    def determinize(self, player_index: int):
        self.gather_informations()
        unseen_cards = self.get_unseen_cards(player_index)
        if not unseen_cards:
            return self.copy()

        random.shuffle(unseen_cards)
        other_players = {p for p in range(4) if p != player_index}

        random_hands = [[] for _ in range(4)]
        pointer = 0
        for p in range(4):
            if p == player_index:
                random_hands[p] = self.hands[p].copy()
            else:
                new_pointer = pointer + len(self.hands[p])
                random_hands[p] = unseen_cards[pointer:new_pointer]  # do i rly have to copy ?
                pointer = new_pointer

        while next_illegal := self.get_next_illegal_cards(random_hands):
            p, card = next_illegal
            possible_trading_players = [tp for tp in other_players if self.can_own(tp, card)]
            random.shuffle(possible_trading_players)
            for trading_player in possible_trading_players:
                for trade_card in random_hands[trading_player]:
                    if self.can_own(p, trade_card) and self.can_own(trading_player, card):
                        random_hands[p][random_hands[p].index(card)] = trade_card
                        random_hands[trading_player][random_hands[trading_player].index(trade_card)] = card
                        break
                else:
                    continue
                break
            else:
                second = [tp for tp in other_players if self.can_own(tp, card)][0]
                third = [tp for tp in other_players if tp != second and tp != p][0]
                assert {p, second, third} == set(other_players)

                for trade_card in random_hands[second]:
                    if self.can_own(third, trade_card):
                        for trade_trade_card in random_hands[third]:
                            if self.can_own(p, trade_trade_card):
                                random_hands[p][random_hands[p].index(card)] = trade_trade_card
                                random_hands[third][random_hands[third].index(trade_trade_card)] = card
                                random_hands[second][random_hands[second].index(trade_card)] = trade_trade_card
                                break
                        else:
                            continue
                        break
                else:
                    print([(tp, self.can_own(p, card)) for tp in other_players])
                    print(get_card_name(card))
                    print(pprint_tricks(self.hands))
                    print(self.info["ruff"])
                    print(pprint_tricks(random_hands))
                    raise ValueError("No possible trade")

        return self.copy(hands=random_hands)

    def get_current_player(self):
        return (self.current_lead + len(self.current_trick)) % 4

    def __repr__(self):
        betting_team = f"{self.names[self.betting_team]}/ {self.names[self.betting_team+2%4]}"
        contract = f'team {betting_team} joue à {self.bet_value}{["♠", "♥", "♦", "♣"][self.trump]}'
        hands = [f"{name}: {pprint_trick([*sorted(hand)])}" for name, hand in zip(self.names, self.hands)]
        passed_tricks = "\n".join(
            [f"{' '.join(get_card_name(card) for card in trick)}" for trick in self.tricks]
            + [" ".join(get_card_name(card) for card in self.current_trick)]
        )
        return "\n---\n".join((contract, "\n".join(hands), passed_tricks, str(self.scores))) + "\n---"

    @lru_cache(maxsize=128)  # could be 1 ?
    def get_legal_actions(self):

        if not self.current_trick:  # also works for already finished game
            return self.hands[self.get_current_player()]

        leading_suit = self.current_trick[0] // 8
        ranks = get_ranks(self.trump, leading_suit)

        hand = self.hands[self.get_current_player()]

        assert hand, f"You have no card in your hand, {self.current_lead} {self.get_current_player()} {self}"

        hand_suits = [card // 8 for card in hand]
        hand_trumps = [card for card in hand if card // 8 == self.trump]

        if leading_suit in hand_suits:  # tu as la couleur demandée
            if leading_suit == self.trump:  # c'est l'atout
                higher_trumps = get_higher_trumps(self.current_trick, hand_trumps, ranks)
                return higher_trumps if higher_trumps else hand_trumps
            else:
                return [card for i, card in enumerate(hand) if hand_suits[i] == leading_suit]

        # tu n'as pas la couleur demandée
        current_trick_ranks = [ranks[card] for card in self.current_trick]
        partner_is_winning = (
            False if len(self.current_trick) == 1 else max(current_trick_ranks) == current_trick_ranks[-2]
        )

        if partner_is_winning:
            return hand

        if hand_trumps:
            higher_trumps = get_higher_trumps(self.current_trick, hand_trumps, ranks)
            return higher_trumps if higher_trumps else hand_trumps

        return hand


def get_higher_trumps(current_trick, hand_trumps, ranks):
    max_trick_trump_value = max(ranks[card] for card in current_trick)
    return [card for card in hand_trumps if ranks[card] > max_trick_trump_value]


@lru_cache(maxsize=None)
def get_ranks(trump, lead):
    ranks = [-1] * 32
    ranks[8 * lead : 8 * lead + 8] = [0, 1, 2, 6, 3, 4, 5, 7]
    ranks[8 * trump : 8 * trump + 8] = [8, 9, 14, 12, 15, 10, 11, 13]
    return ranks


@lru_cache(maxsize=None)
def get_scores(trump):
    scores = [0, 0, 0, 10, 2, 3, 4, 11] * 4
    scores[8 * trump : 8 * trump + 8] = [0, 0, 14, 10, 20, 3, 4, 11]
    return scores


@lru_cache(maxsize=None)
def get_card_name(index):
    values = ["7", "8", "9", "10", "J", "Q", "K", "A"]
    suits = ["♠", "♥", "♦", "♣"]
    return f"{values[index % 8] + suits[index // 8]:>3}"


def get_cards_names(list_of_cards):
    return " ".join(get_card_name(card) for card in list_of_cards)


def get_belote_rebelote(game_state):
    queen, king = BELOTE_REBELOTE[game_state.trump]
    for i, original_hand in enumerate(map(list, zip(*game_state.tricks))):
        if queen in original_hand and king in original_hand:
            return i % 2


def final_points(game_state: GameState):
    "update game_state object in place"
    belote_reblote_team = get_belote_rebelote(game_state)
    if belote_reblote_team is not None:
        game_state.scores[belote_reblote_team] += 20
    game_state.scores[game_state.current_lead % 2] += 10  # 10 de der

    # contrat rempli
    if game_state.scores[game_state.betting_team] >= game_state.bet_value:
        game_state.scores[game_state.betting_team] += game_state.bet_value * game_state.coinche

    # contrat manqué
    else:
        game_state.scores = [0, 0]
        game_state.scores[game_state.betting_team ^ 1] += 160 + (game_state.bet_value * game_state.coinche)
        if belote_reblote_team is not None:
            game_state.scores[belote_reblote_team] += 20


def play_one_game(agents: list, game_state: GameState, verbose=False):
    assert len(agents) == 4 and not game_state.tricks

    while len(game_state.tricks) < 8:
        idx = game_state.get_current_player()
        card = agents[idx].play(game_state.copy())
        play_one_card(card, game_state, verbose)
        if verbose and not game_state.current_trick:
            print(game_state.scores)
            print("-" * 30)


def play_one_card(card, game_state: GameState, verbose=False):
    "update game_state object in place"
    assert len(game_state.current_trick) < 4 and len(game_state.tricks) < 8

    idx = game_state.get_current_player()
    assert card in game_state.hands[idx], f"{game_state.names[idx]} tried to play {get_card_name(card)}"

    game_state.current_trick.append(card)
    game_state.hands[idx].remove(card)
    game_state.last_card_played = card

    if verbose:
        print(
            f"{get_card_name(card)} played by {game_state.names[idx]}. Current trick is {[get_card_name(c) for c in game_state.current_trick]}"
        )
        print("-" * 30)

    if len(game_state.current_trick) == 4:
        scores = get_scores(game_state.trump)
        ranks = get_ranks(game_state.trump, game_state.current_trick[0] // 8)
        winner_card = max(game_state.current_trick, key=lambda x: ranks[x])
        indexed_trick = {card: i % 4 for card, i in zip(game_state.current_trick, range(idx + 1, idx + 5))}
        current_trick_ordered = list(sorted(indexed_trick, key=lambda x: indexed_trick[x]))

        game_state.scores[indexed_trick[winner_card] % 2] += sum(scores[card] for card in game_state.current_trick)

        game_state.leads.append(game_state.current_lead)
        game_state.current_lead = indexed_trick[winner_card]
        game_state.tricks.append(current_trick_ordered)
        game_state.current_trick = []

        if len(game_state.tricks) == 8:
            final_points(game_state)


class Agent:
    def __init__(self, name="Agent"):
        self.name = name

    def play(self, game_state: GameState):
        raise NotImplementedError

    def bid(self, game_state: GameState):
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, name="Randy"):
        super().__init__(name=name)

    def play(self, game_state: GameState):
        return random.choice(game_state.get_legal_actions())

    def bid(self, hand, bids):
        if any(bid[1] is not None for bid in bids):
            return None, None
        else:
            return 80, random.randint(0, 3)


class HumanAgent(Agent):
    def __init__(self, name="Randy"):
        super().__init__(name=name)

    def play(self, game_state: GameState):

        print("-" * 3)
        legal_actions = game_state.get_legal_actions()

        if len(legal_actions) == 1:
            print(f"Only legal card {get_card_name(legal_actions[0])} was played")
            return legal_actions[0]

        print(f"It's {self.name}'s turn")
        print(f"Legal actions: {get_cards_names(legal_actions)}")
        print(f"card ints    : {' '.join([f'{c:>3}' for c in legal_actions])}")
        card = int(input("Choose a card: "))

        while card not in legal_actions:
            print("This card is not legal")
            card = int(input("Choose a card: "))

        return card

    def bid(self, hand, bids):
        print(f"It's {self.name}'s turn")
        print("Choose a bid")
        value = input("Value: ")
        trump = input("Trump: ")
        if value != "pass":
            return int(trump), int(value)
        return None, None


def ucb1(parent, child, parent_visit_log, temp, maximum_score):
    exploitation = (child.value / maximum_score) / child.visits
    exploration = math.sqrt(parent_visit_log / child.visits)
    return exploitation + temp * exploration


class Node:
    def __init__(self, move=None, parent=None, last_player=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.last_player = last_player

    def __repr__(self):
        return f"Node(visits={self.visits:>6}, relative_value= {self.value/162/self.visits:.2f}, predicted_score={self.value/self.visits if self.visits else 0:.0f})"

    def select(self, legal_actions, temp=0.7, maximum_score=300):
        parent_visit_log = math.log(self.visits)
        legal_children = [child for child in self.children if child.move in legal_actions]
        ucb1_scores = [ucb1(self, child, parent_visit_log, temp, maximum_score) for child in legal_children]
        return legal_children[ucb1_scores.index(max(ucb1_scores))]

    def get_untried_moves(self, legal_actions):
        tried_moves = [child.move for child in self.children]
        return [action for action in legal_actions if action not in tried_moves]

    def add_child(self, move, last_player):
        child = Node(move=move, parent=self, last_player=last_player)
        self.children.append(child)
        return child

    def update(self, state):
        self.visits += 1
        if self.last_player is not None:
            self.value += state.scores[self.last_player % 2]


class OracleAgent(Agent):
    def __init__(self, name="Oracle", iterations=1000, verbose=False):
        super().__init__(name=name)
        self.iterations = iterations
        self.verbose = verbose
        self.player_index = None
        self.predicted_scores = []

    def play(self, game_state: GameState):

        # for the score index
        self.player_index = game_state.get_current_player()

        legal_actions = game_state.get_legal_actions()

        if len(legal_actions) == 1:
            if self.verbose:
                print(f"{self.name}: only legal card {get_card_name(legal_actions[0])} was played")
            return legal_actions[0]

        root = Node(game_state.copy(), parent=None)

        for _ in range(self.iterations):
            leaf = self.traverse(root)  # Selection
            simulation_results = self.rollout(leaf.game_state.copy())  # Simulation
            self.backpropagate(leaf, simulation_results)  # Backpropagation

        if self.verbose:
            print(f"{self.name}: {_ + 1} iterations")
            for child in root.children:
                print("\t", get_card_name(child.game_state.last_card_played), child)

        return self.best_move(root)

    def traverse(self, node, state, maximum_score=300):
        while len(state.tricks) < 8:
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.select(maximum_score=maximum_score)
        return node

    def expand(self, node):
        legal_actions = node.game_state.get_legal_actions()
        for action in legal_actions:
            if not any(action == child.game_state.last_card_played for child in node.children):
                new_state = node.game_state.copy()
                current_player_index = new_state.get_current_player()
                play_one_card(action, new_state)
                new_node = Node(new_state, parent=node, last_player_team=current_player_index % 2)
                node.children.append(new_node)
                return new_node
        raise Exception("Should not reach here")

    def rollout(self, game_state: GameState):
        while len(game_state.tricks) < 8:
            possible_actions = game_state.get_legal_actions()
            action = random.choice(possible_actions)
            play_one_card(action, game_state)
        return game_state.scores

    def backpropagate(self, node, results):
        while node:
            node.visits += 1
            if node.last_player_team is not None:
                node.value += results[node.last_player_team]
            node = node.parent

    def best_move(self, node):
        best_move_node = max(node.children, key=lambda x: x.visits)
        last_card_played = best_move_node.game_state.last_card_played
        return last_card_played

    def estimate_score(self, game_state: GameState):

        root = Node(None, parent=None)
        hand = game_state.hands[self.player_index]
        unseen_cards = [c for c in range(32) if c not in hand]

        for _ in range(self.iterations // 4):

            possible_start = game_state.copy()
            _unseen_cards = unseen_cards.copy()

            while possible_start.get_current_player() != self.player_index:
                card = random.choice(possible_start.get_legal_actions())
                play_one_card(card, possible_start)
                _unseen_cards.remove(card)

            root.game_state = possible_start

            leaf = self.traverse(root)  # Selection
            simulation_result = self.rollout(leaf.game_state.copy())  # Simulation
            self.backpropagate(leaf, simulation_result)  # Backpropagation

        # if self.verbose:
        #     print(f"{self.name}: {_ + 1:_} iterations")
        #     for child in sorted(root.children, key=lambda x: x.visits, reverse=True):
        #         print("\t", get_card_name(child.game_state.last_card_played), child)

        best_move_node = max(root.children, key=lambda x: x.visits)
        return round(best_move_node.value / best_move_node.visits)

    def bid(self, hands, current_lead, bids):
        if not self.predicted_scores:
            player_idx = (current_lead + len(bids)) % 4
            self.player_index = player_idx

            for potential_trump in range(4):

                if self.verbose:
                    print(self.name, f"thinking about announcing {SUITS[potential_trump]}")

                game_state = GameState(
                    names=["N", "E", "S", "W"],
                    bids=[],
                    bet_value=0,
                    betting_team=player_idx % 2,
                    trump=potential_trump,
                    coinche=1,
                    hands=[h.copy() for h in hands],
                    tricks=[],
                    leads=[],
                    current_trick=[],
                    current_lead=current_lead,
                    last_card_played=None,
                    scores=[0, 0],
                    info=get_empty_info_dict(),
                )

                self.predicted_scores.append(self.estimate_score(game_state.copy()))

            if self.verbose:
                print(
                    f"{self.name}: best I can do is {[f'{score}{SUITS[t]}' for t, score in enumerate(self.predicted_scores)]}"
                )

        best_bid = max(enumerate(self.predicted_scores), key=lambda x: x[1])
        best_bid = (best_bid[0], best_bid[1] // 10 * 10)

        last_bid_value = 0
        for bid in bids[::-1]:
            if bid[1] is not None:
                last_bid_value = bid[1]
                break

        if best_bid[1] > last_bid_value:
            return best_bid
        else:
            # TODO: implementer une défense selon best bid
            return None, None


class DuckAgent(OracleAgent):
    def __init__(self, name="Duck", iterations=10_000, verbose=False):
        super().__init__(name=name, iterations=iterations, verbose=verbose)
        self.predicted_scores = []

    def play(self, root_state: GameState):

        legal_actions = root_state.get_legal_actions()

        if len(legal_actions) == 1:
            if self.verbose:
                print(f"{self.name}: only legal card {get_card_name(legal_actions[0])} was played")
            return legal_actions[0]

        root_node = Node(None, parent=None)

        for _ in range(self.iterations):

            node = root_node

            # Determine
            state = root_state.determinize(self.player_index)  # determinize

            # Select
            while len(state.tricks) < 8 and not node.get_untried_moves(state.get_legal_actions()):
                node = node.select(state.get_legal_actions())
                play_one_card(node.move, state)

            # Expand
            untried_moves = node.get_untried_moves(state.get_legal_actions())
            if untried_moves:
                move = random.choice(untried_moves)
                current_player = state.get_current_player()
                play_one_card(move, state)
                node = node.add_child(move=move, last_player=current_player)

            # Simulate
            while len(state.tricks) < 8:
                possible_actions = state.get_legal_actions()
                action = random.choice(possible_actions)
                play_one_card(action, state)

            # Backpropagate
            while node:
                node.update(state)
                node = node.parent

        if self.verbose:
            print(f"{self.name}: {_ + 1:_} iterations")
            for child in sorted(root_node.children, key=lambda x: x.visits, reverse=True):
                print("\t", get_card_name(child.move), child)

        best_move_node = max(root_node.children, key=lambda x: x.visits)
        return best_move_node.move

    def estimate_score(self, root_state: GameState):

        root_node = Node(None, parent=None)

        for _ in range(self.iterations):

            node = root_node

            # Determine
            state = root_state.determinize(self.player_index)  # determinize

            # Quick fix-hack TODO: corriger
            state.current_lead = self.player_index

            # Select
            while len(state.tricks) < 8 and not node.get_untried_moves(state.get_legal_actions()):
                node = node.select(state.get_legal_actions())
                play_one_card(node.move, state)

            # Expand
            untried_moves = node.get_untried_moves(state.get_legal_actions())
            if untried_moves:
                move = random.choice(untried_moves)
                current_player = state.get_current_player()
                play_one_card(move, state)
                node = node.add_child(move=move, last_player=current_player)

            # Simulate
            while len(state.tricks) < 8:
                possible_actions = state.get_legal_actions()
                action = random.choice(possible_actions)
                play_one_card(action, state)

            # Backpropagate
            while node:
                node.update(state)
                node = node.parent

        best_move_node = max(root_node.children, key=lambda x: x.visits)

        if self.verbose:
            print(f"{self.name}: {_ + 1:_} iterations")
            for child in sorted(root_node.children, key=lambda x: x.visits, reverse=True):
                print("\t", get_card_name(child.move), child)
            # print(f"children of {get_card_name(best_move_node.move)}")
            # for child in best_move_node.children:
            #     print("\t", get_card_name(child.move), child)

        return round(best_move_node.value / best_move_node.visits)

    def bid(self, hand, current_lead, bids):
        if not self.predicted_scores:
            player_idx = (current_lead + len(bids)) % 4
            self.player_index = player_idx

            unseen_cards = [c for c in range(32) if c not in hand]
            random.shuffle(unseen_cards)
            hands = [unseen_cards[i : i + 8] for i in (0, 8, 16)]
            hands.insert(player_idx, hand)

            for potential_trump in range(4):
                if self.verbose:
                    print(self.name, f"thinking about announcing {SUITS[potential_trump]}")
                game_state = GameState(
                    names=["N", "E", "S", "W"],
                    bids=[],
                    bet_value=80,
                    betting_team=self.player_index % 2,
                    trump=potential_trump,
                    coinche=1,
                    hands=[h.copy() for h in hands],
                    tricks=[],
                    leads=[],
                    current_trick=[],
                    current_lead=current_lead,
                    last_card_played=None,
                    scores=[0, 0],
                    info=get_empty_info_dict(),
                )
                self.predicted_scores.append(max(0, self.estimate_score(game_state) - 80))

            if self.verbose:
                print(
                    f"{self.name}: best I can do is {[f'{score}{SUITS[t]}' for t, score in enumerate(self.predicted_scores)]}"
                )

        best_bid = max(enumerate(self.predicted_scores), key=lambda x: x[1])
        best_bid = (best_bid[0], best_bid[1] // 10 * 10)

        last_bid_value = 70
        for bid in bids[::-1]:
            if bid[1] is not None:
                last_bid_value = bid[1]
                break

        if best_bid[1] > last_bid_value:
            return best_bid
        else:
            # TODO: implementer une défense selon best bid
            return None, None


def bidding_phase(players, current_lead, hands, verbose):

    bids = []
    best_bid = 1

    while len(bids) < 4 or any(bid[1] for bid in bids[-3:]):
        player = (current_lead + len(bids)) % 4

        if isinstance(players[player], DuckAgent):
            trump, value = players[player].bid(hands[player], current_lead, bids)
        else:  # all hands
            trump, value = players[player].bid(hands, current_lead, bids)

        assert (
            not value or value > best_bid
        ), f"{players[player].name} tried to bid {value} while best bid is {best_bid}"
        best_bid = value or best_bid

        bids.append([player, value, trump])

        if verbose:
            if value:
                print(f"{players[player].name} bids {value} {SUITS[trump]}")
            else:
                print(f"{players[player].name} passes")

    return bids[:-3]


def main():

    # 46 - 110 carreau - team 2
    # 23 - 90 coeur - team 1

    n_iter = 1000
    big_scores = [0, 0]

    for i in range(1):

        # randys = [
        #     RandomAgent(name="Jean"),
        #     RandomAgent(name="Ivan"),
        #     RandomAgent(name="Jules"),
        #     RandomAgent(name="Eloi"),
        # ]

        # players = [
        #     OracleAgent(name="Jean", iterations=n_iter, verbose=True),
        #     # HumanAgent(name="Jean"),
        #     OracleAgent(name="Ivan", iterations=n_iter, verbose=True),
        #     # DuckAgent(name="Ivan", iterations=n_iter, verbose=True),
        #     # DuckAgent(name="Ivan", iterations=n_iter, verbose=True),
        #     # HumanAgent(name="Gaspard"),
        #     OracleAgent(name="Jule", iterations=n_iter, verbose=True),
        #     # HumanAgent(name="Jule"),
        #     OracleAgent(name="Eloi", iterations=n_iter, verbose=True),
        #     # DuckAgent(name="Eloi", iterations=n_iter, verbose=True),
        #     # HumanAgent(name="Roland"),
        # ]

        players = [
            DuckAgent(name="Jean", iterations=n_iter, verbose=True),
            # RandomAgent(name="Ivan"),
            DuckAgent(name="Ivan", iterations=n_iter, verbose=True),
            DuckAgent(name="Jule", iterations=n_iter, verbose=True),
            DuckAgent(name="Eloi", iterations=n_iter, verbose=True),
            # RandomAgent(name="Eloi"),
        ]

        # print(i)
        game_state = GameState.fresh_game(
            players=players,
            # hands=[[*range(i, i + 8)] for i in [0, 8, 16, 24]],
            # seed=i,
            verbose=True,
        )

        # print(game_state)
        # game_state2 = GameState.fresh_game(
        #     names=["Ivan", "Jean", "Eloi", "Jules"], bet_value=110, betting_team=1, trump=2, coinche=1, seed=46
        # )

        play_one_game(players, game_state, verbose=True)

        big_scores[0] += game_state.scores[0]
        big_scores[1] += game_state.scores[1]

        # for _ in range(3):
        #     for i in range(4):
        #         card = ducks[i].play(game_state.copy())
        #         play_one_card(card, game_state, verbose=True)
        #     print(game_state.scores)
        #     print("-" * 30)

        # for i in range(2):
        #     card = AGENTS[i].play(game_state)
        #     play_one_card(card, game_state, verbose=True)

        # print(game_state)
        # print("-" * 50)
        # game_state.gather_informations()
        # unseen_cards = game_state.get_unseen_cards(0)
        # deter = game_state.determinize(0, unseen_cards)
        # print(pprint_tricks(deter.hands))
        # print("-" * 50)

        # print(game_state)
        # print(game_state.scores)

    print(big_scores)


if __name__ == "__main__":

    # main()

    # PROFILE
    pr = cProfile.Profile()
    pr.enable()
    main()  # Call the main function where your code is executed
    pr.disable()
    pr.dump_stats("profile_results.prof")
