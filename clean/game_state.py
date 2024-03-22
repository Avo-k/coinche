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
SUITS = '♠♥♦♣'


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

        assert not value or value > best_bid, f"{players[player].name} tried to bid {value} while best bid is {best_bid}"
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
        verbose=False,
    ):

        assert len(players) == 4, "There must be 4 players"

        if seed is not None:
            random.seed(seed)

        deck = list(range(32))
        random.shuffle(deck)

        hands = [deck[i * 8 : i * 8 + 8] for i in range(4)]

        if verbose:
            for p, h in zip(players, hands):
                print(f"{p.name}: {pprint_trick(h)}")

        bids, belote_rebelote = bidding_phase(players, current_lead, hands, verbose=True)
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
        if not unseen_cards: # last_card
            return self.copy()

        # SUR
        # si un joueur coupe ou pisse, il n'a plus la couleur
        # si un joueur pisse et que partenaire pas maitre, il n'a pas d'atout
        # si un joueur ne monte pas à l'atout alors qu'il devrait, il n'a pas de plus gros atout que le précédent

        # PROBABLE
        # si un joueur a dit belote, il a l'autre carte de la paire roi dame
        # si le joueur annonce une couleur il a au moins 1 carte de cette couleur

        random_hands = [[] for _ in range(4)]
        random_hands[player_index] = self.hands[player_index].copy()

        players = list(range(4))
        players.remove(player_index)

        def not_ruff(player, card):
            return not self.info["ruff"][player][card // 8]

        steps = []

        while unseen_cards:
            # is there a card that is only possible for one player?
            could_be_their_card = {c: [p for p in players if not_ruff(p, c)] for c in unseen_cards}
            for card in unseen_cards:
                if len(could_be_their_card[card]) == 1:
                    next_player = could_be_their_card[card][0]
                    break
            else: # choose the player with the least possible cards
                complexity_order = {p: [c for c in unseen_cards if not_ruff(p, c)] for p in players}
                if not players:
                    print(player_index)
                    print(self.hands)
                    print(unseen_cards)
                    quit()
                next_player = min(players, key=lambda x: len(complexity_order[x]))
                card = random.choice(complexity_order[next_player])

            random_hands[next_player].append(card)
            unseen_cards.remove(card)
            if len(random_hands[next_player])==len(self.hands[next_player]):
                players.remove(next_player)

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

    def get_current_player(self):
        return (self.current_lead + len(self.current_trick)) % 4

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


def get_legal_actions(gs: GameState):

    assert len(gs.tricks) < 8, "The game is already over"

    if not gs.current_trick:  # no card played
        return gs.hands[gs.current_lead]

    leading_suit = gs.current_trick[0] // 8
    ranks = get_ranks(gs.trump, leading_suit)

    hand = gs.hands[gs.get_current_player()]

    assert hand, "You have no card in your hand"

    hand_suits = [card // 8 for card in hand]
    hand_trumps = [card for card in hand if card // 8 == gs.trump]

    if leading_suit in hand_suits:  # tu as la couleur demandée
        if leading_suit == gs.trump:  # atout demandé
            if hand_trumps:
                higher_trumps = get_higher_trumps(gs.current_trick, hand_trumps, ranks)
                return higher_trumps if higher_trumps else hand_trumps
            else:  # patatou
                return hand
        else:
            return [card for i, card in enumerate(hand) if hand_suits[i] == leading_suit]

    # tu n'as pas la couleur demandée
    current_trick_ranks = [ranks[card] for card in gs.current_trick]
    partner_is_winning = False if len(gs.current_trick) == 1 else max(current_trick_ranks) == current_trick_ranks[-2]

    if partner_is_winning:
        return hand

    if hand_trumps:
        higher_trumps = get_higher_trumps(gs.current_trick, hand_trumps, ranks)
        return higher_trumps if higher_trumps else hand_trumps

    return hand


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


def final_points(game_state: GameState):
    "update game_state object in place"
    if game_state.belote_rebelote:
        game_state.scores[game_state.belote_rebelote] += 20
    game_state.scores[game_state.current_lead % 2] += 10  # 10 de der

    # contrat rempli
    if game_state.scores[game_state.betting_team] >= game_state.bet_value:
        game_state.scores[game_state.betting_team] += game_state.bet_value * game_state.coinche

    # contrat manqué
    else:
        game_state.scores = [0, 0]
        game_state.scores[game_state.betting_team ^ 1] += 160 + (game_state.bet_value * game_state.coinche)
        if game_state.belote_rebelote:
            game_state.scores[game_state.belote_rebelote] += 20


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
        idx += 1
        scores = get_scores(game_state.trump)
        ranks = get_ranks(game_state.trump, game_state.current_trick[0] // 8)
        winner_card = max(game_state.current_trick, key=lambda x: ranks[x])
        indexed_trick = {card: idx % 4 for card, idx in zip(game_state.current_trick, range(idx, idx + 4))}
        ordered_current_trick = list(sorted(game_state.current_trick, key=indexed_trick.get))

        game_state.scores[indexed_trick[winner_card] % 2] += sum(scores[card] for card in game_state.current_trick)

        game_state.leads.append(game_state.current_lead)
        game_state.current_lead = indexed_trick[winner_card]
        game_state.tricks.append(ordered_current_trick)
        game_state.current_trick = []

        if len(game_state.tricks) == 8:
            final_points(game_state)


def play_n_random_cards(n: int, game_state: GameState, verbose=False):
    "update game_state object in place"
    while n and len(game_state.tricks) < 8:
        card = random.choice(get_legal_actions(gs=game_state))
        play_one_card(card, game_state, verbose=verbose)
        n -= 1



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
        return random.choice(get_legal_actions(gs=game_state))
    
    def bid(self, hand, bids):
        if any(bid[1] is not None for bid in bids):
            return None, None
        else:
            return 80, random.randint(0, 3)


class HumanAgent(Agent):
    def __init__(self, name="Randy"):
        super().__init__(name=name)

    def play(self, game_state: GameState):
        
        print("-"*3)
        legal_actions = get_legal_actions(gs=game_state)

        if len(legal_actions) == 1:
            print(f"Only legal card {get_card_name(legal_actions[0])} was played")
            return legal_actions[0]

        print(f"It's {self.name}'s turn")
        print(f"Legal actions: {get_cards_names(legal_actions)}")
        print(f"card ints    : {" ".join([f"{c:>3}" for c in legal_actions])}")
        card = int(input("Choose a card: "))

        while card not in legal_actions:
            print("This card is not legal")
            card = int(input("Choose a card: "))

        return card

    def bid(self, game_state: GameState):
        print(f"It's {self.name}'s turn")
        print("Choose a bid")
        value = int(input("Value: "))
        trump = int(input("Trump: "))
        return value, trump
    

def ucb1(parent, child, parent_visit_log, temp=3):
    exploitation = child.value / 200 / child.visits
    exploration = math.sqrt(parent_visit_log / child.visits)
    return exploitation + temp * exploration


class Node:
    def __init__(self, game_state: GameState, parent=None):
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.is_fully_expanded = False

    def __repr__(self):
        return f"Node(visits={self.visits:_<6}, predicted_score={self.value/self.visits if self.visits else 0:.0f})"

    def is_terminal(self):
        return len(self.game_state.tricks) == 8

    def select(self, temp=2):
        parent_visit_log = math.log(self.visits)
        ucb1_scores = [ucb1(self, child, parent_visit_log, temp) for child in self.children]
        return self.children[ucb1_scores.index(max(ucb1_scores))]


class OracleAgent(Agent):
    def __init__(self, name="Oracle", iterations=1000, verbose=False):
        super().__init__(name=name)
        self.iterations = iterations
        self.verbose = verbose
        self.player_index = None

    def play(self, game_state: GameState):

        # for the score index
        self.player_index = game_state.get_current_player()

        legal_actions = get_legal_actions(game_state)

        if len(legal_actions) == 1:
            if self.verbose:
                print(f"{self.name}: only legal card {get_card_name(legal_actions[0])} was played")
            return legal_actions[0]

        root = Node(game_state.copy(), parent=None)

        for _ in range(self.iterations):
            
            leaf = self.traverse(root)  # Selection
            simulation_result = self.rollout(leaf.game_state.copy())  # Simulation
            self.backpropagate(leaf, simulation_result)  # Backpropagation

        if self.verbose:
            print(f"{self.name}: {_ + 1} iterations")
            for child in root.children:
                print("\t", get_card_name(child.game_state.last_card_played), child)

        return self.best_move(root)

    def traverse(self, node):
        while not node.is_terminal():
            if not node.is_fully_expanded:
                exp = self.expand(node)
                if exp: # this is a hack, not sure if correct
                    return exp
                node = node.select()
            else:
                node = node.select()
        return node

    def expand(self, node):
        assert (
            sum(1 for hand in node.game_state.hands for _ in hand)
            + sum(1 for trick in node.game_state.tricks for _ in trick)
            + len(node.game_state.current_trick)
            == 32
        ), f"Some cards probably fell on the floor... {node.game_state}"

        legal_actions = get_legal_actions(node.game_state)
        assert legal_actions, "No legal action"

        for action in legal_actions:
            if not any(action == child.game_state.last_card_played for child in node.children):
                new_state = node.game_state.copy()
                play_one_card(action, new_state)
                new_node = Node(new_state, parent=node)
                node.children.append(new_node)
                if len(legal_actions) == len(node.children):
                    node.is_fully_expanded = True
                return new_node
        # raise Exception("Should not reach here")

    def rollout(self, game_state: GameState):
        while len(game_state.tricks) < 8:
            possible_actions = get_legal_actions(game_state)
            action = random.choice(possible_actions)
            play_one_card(action, game_state)
        return game_state.scores[self.player_index % 2]

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent

    def best_uct(self, node):
        choices_weights = [ucb1(node, child) for child in node.children]
        return node.children[choices_weights.index(max(choices_weights))]

    def best_move(self, node):
        best_move_node = max(node.children, key=lambda x: x.visits)
        last_card_played = best_move_node.game_state.last_card_played

        assert 0 <= last_card_played < 32, f"{self.name} tried to play {last_card_played} which is not a card"
        assert last_card_played in get_legal_actions(node.game_state), f"{self.name} tried to play {get_card_name(last_card_played)} which is not in their legal cards"

        return last_card_played


def pretend_ownership(game_state: GameState, player_index: int, card:int):
    "modify in place a game state where the player has a given card in their hand"
    
    for p in range(4):
        if card in game_state.hands[p]:
            if p == player_index:
                return
            replacement = game_state.hands[player_index][0]
            game_state.hands[player_index][0] = card
            game_state.hands[p][game_state.hands[p].index(card)] = replacement


class DuckAgent(OracleAgent):
    def __init__(self, name="Duck", iterations=10_000, verbose=False):
        super().__init__(name=name, iterations=iterations, verbose=verbose)
        self.predicted_scores = []

    def play(self, game_state: GameState):

        legal_actions = get_legal_actions(game_state)

        if len(legal_actions) == 1:
            if self.verbose:
                print(f"{self.name}: only legal card {get_card_name(legal_actions[0])} was played")
            return legal_actions[0]

        root = Node(None, parent=None)

        # update game_state.info
        game_state.gather_informations()
        unseen_cards = game_state.get_unseen_cards(self.player_index) 

        if len(game_state.tricks) > 5:
            iterations = min(self.iterations, 1_000*(8-len(game_state.tricks)))
        else:
            iterations = self.iterations

        for _ in range(iterations):

            predicted_game_state = game_state.determinize(
                self.player_index, unseen_cards=unseen_cards.copy()
            )  # determinize
            root.game_state = predicted_game_state

            leaf = self.traverse(root)  # Selection
            if not _:  # first iteration
                assert leaf.game_state.last_card_played in legal_actions
            simulation_result = self.rollout(leaf.game_state.copy())  # Simulation
            self.backpropagate(leaf, simulation_result)  # Backpropagation

        if self.verbose:
            print(f"{self.name}: {_ + 1} iterations")
            for child in root.children:
                print("\t", get_card_name(child.game_state.last_card_played), child)

        return self.best_move(root)

    def estimate_score(self, game_state: GameState):

        root = Node(None, parent=None)
        hand = game_state.hands[self.player_index]
        unseen_cards = [c for c in range(32) if c not in hand]

        for _ in range(self.iterations // 4):

            possible_start = game_state.copy()
            _unseen_cards = unseen_cards.copy()

            if possible_start.get_current_player() != self.player_index:
            
                random.shuffle(_unseen_cards)
                random_hands = [_unseen_cards[i : i + 8] for i in (0, 8, 16)]
                random_hands.insert(self.player_index, hand)

                possible_start.hands = random_hands

                while possible_start.get_current_player() != self.player_index:
                    card = random.choice(get_legal_actions(possible_start))
                    play_one_card(card, possible_start)
                    _unseen_cards.remove(card)

            predicted_game_state = possible_start.determinize(
                self.player_index, unseen_cards=_unseen_cards.copy()
            )  # determinize
            root.game_state = predicted_game_state

            # print(root.children)

            leaf = self.traverse(root)  # Selection
            simulation_result = self.rollout(leaf.game_state.copy())  # Simulation
            self.backpropagate(leaf, simulation_result)  # Backpropagation

        # if self.verbose:
        #     print(f"{self.name}: {_ + 1} iterations")
        #     for child in root.children:
        #         print("\t", get_card_name(child.game_state.last_card_played), child)

        # for c in root.children:
        #     print(get_card_name(c.game_state.last_card_played), c)

        best_move_node = max(root.children, key=lambda x: x.visits)
        return round(best_move_node.value / best_move_node.visits)


    def bid(self, hand, bids):

        if not self.predicted_scores:

            current_lead = bids[0][0] if bids else 0
            player_idx = (current_lead + len(bids)) % 4
            self.player_index = player_idx

            unseen_cards = [c for c in range(32) if c not in hand]
            assert len(unseen_cards) == 24
            random.shuffle(unseen_cards)
            hands = [unseen_cards[i : i + 8] for i in (0, 8, 16)]
            hands.insert(player_idx, hand)

            assert list(sorted(sum(hands, []))) == list(range(32)), str(hands)

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
                    belote_rebelote=None,
                    last_card_played=-1,
                    scores=[0, 0],
                    info=get_empty_info_dict(),
                )
                # while (len(game_state.current_trick) + game_state.current_lead) % 4 != player_idx:
                #     card = random.choice(get_legal_actions(game_state))
                #     play_one_card(card, game_state)

                self.predicted_scores.append((self.estimate_score(game_state.copy()) - 50))

            if self.verbose:
                print(f"{self.name}: best I can do is {[f'{score}{SUITS[t]}' for t, score in enumerate(self.predicted_scores)]}")

        best_bid = max(enumerate(self.predicted_scores), key=lambda x: x[1])//10*10

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


def main():

    # 46 - 110 carreau - team 2
    # 23 - 90 coeur - team 1

    n_iter = 1_000_000

    for i in range(1):

        # randys = [
        #     RandomAgent(name="Jean"),
        #     RandomAgent(name="Ivan"),
        #     RandomAgent(name="Jules"),
        #     RandomAgent(name="Eloi"),
        # ]

        # oracles = [
        #     OracleAgent(name="Jean", iterations=n_iter, verbose=True),
        #     # HumanAgent(name="Jean"),
        #     # OracleAgent(name="Ivan", iterations=n_iter, verbose=True),
        #     DuckAgent(name="Ivan", iterations=n_iter, verbose=True),
        #     # HumanAgent(name="Ivan"),
        #     OracleAgent(name="Jule", iterations=n_iter, verbose=True),
        #     # HumanAgent(name="Jule"),
        #     # OracleAgent(name="Eloi", iterations=n_iter, verbose=True),
        #     DuckAgent(name="Eloi", iterations=n_iter, verbose=True),
        #     # HumanAgent(name="Eloi"),
        # ]

        ducks = [
            DuckAgent(name="Jean", iterations=n_iter, verbose=True),
            DuckAgent(name="Ivan", iterations=n_iter, verbose=True),
            DuckAgent(name="Jule", iterations=n_iter, verbose=True),
            DuckAgent(name="Eloi", iterations=n_iter, verbose=True),
        ]

        game_state = GameState.fresh_game(
            players=ducks,
            current_lead=0,
            # seed=543536635,
            verbose=True,
        )
        
        print(game_state)

        # game_state2 = GameState.fresh_game(
        #     names=["Ivan", "Jean", "Eloi", "Jules"], bet_value=110, betting_team=1, trump=2, coinche=1, seed=46
        # )

        play_one_game(ducks, game_state, verbose=True)

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
