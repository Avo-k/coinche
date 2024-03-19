import cProfile
import io
import math
import pstats
import random
from functools import lru_cache


BELOTE_REBELOTE = {trump: (5 + 8 * trump, 6 + 8 * trump) for trump in range(4)}


class GameState:
    __slots__ = [
        "names",
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
    ]

    def __init__(
        self,
        names: list[str],
        bet_value: int,  # 80 to 180
        betting_team: int,  # 0: 1st and 3rd players, 1: 2nd and 4th players
        trump: int,  # 0: spades, 1: hearts, 2: diamonds, 3: clubs
        coinche: int,  # 1: no coinche, 2: coinche, 3: surcoinche
        hands: list[list[int]],  # 4 lists of 0 to 8 cards
        tricks: list[list[int]],  # 0 to 8 lists of 4 cards
        leads: list[int],  # 0 to 3 player indices
        current_trick: list[int],  # 0 to 4 cards
        current_lead: int,  # player index, 0 to 3
        belote_rebelote: int,  # None or team index
        last_card_played: int,  # 0 to 31, -1 for no card
        scores: list[int],  # both scores
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

        if belote_rebelote is None:
            self.belote_rebelote = None

    @classmethod
    def fresh_game(
        cls,
        names: list[str] = ["Ivan", "Jean", "Eloi", "Jules"],
        bet_value=90,
        betting_team=0,
        trump=1,
        coinche=1,
        seed=None,
    ):

        assert len(names) == 4, "There must be 4 players"

        if seed is not None:
            random.seed(seed)

        deck = list(range(32))
        random.shuffle(deck)

        return cls(
            names=names,
            bet_value=bet_value,
            betting_team=betting_team,
            trump=trump,
            coinche=coinche,
            hands=[deck[i * 8 : i * 8 + 8] for i in range(4)],
            tricks=[],
            leads=[],
            current_trick=[],
            current_lead=0,
            belote_rebelote=None,
            last_card_played=-1,
            scores=[0, 0],
        )

    def copy(self):
        # Perform a deep copy of the game state. Adjust according to your needs.
        return GameState(
            names=self.names.copy(),
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
        )

    def get_belote_rebelote(self):
        dame, roi = BELOTE_REBELOTE[self.trump]
        for i in range(4):
            if dame in self.hands[i] and roi in self.hands[i]:
                return i % 2
        return None

    @lru_cache(maxsize=None)
    def get_unseen_cards(self, player_index: int):
        return list(set(range(32)).difference(*self.tricks, self.hands[player_index], self.current_trick))

    def determinize(self, player_index: int):
        "Return a potential version of the game state where all information that a given player cannot know is randomized"

        unseen_cards = self.get_unseen_cards(player_index)
        random.shuffle(unseen_cards)

        # SUR
        # si un joueur coupe ou pisse, il n'a plus la couleur
        # si un joueur pisse et que partenaire pas maitre, il n'a pas d'atout
        # si un joueur ne monte pas à l'atout alors qu'il devrait, il n'a pas de plus gros atout que le précédent

        # PROBABLE
        # si un joueur a dit belote, il a l'autre carte de la paire roi dame
        # si le joueur annonce une couleur il a au moins 1 carte de cette couleur

        random_hands = []
        unseen_index = 0

        for i, hand in enumerate(self.hands):
            if i == player_index:
                random_hands.append(hand)
            else:
                hand_size = len(hand)
                random_hands.append(unseen_cards[unseen_index : unseen_index + hand_size])
                unseen_index += hand_size

        return GameState(
            names=self.names.copy(),
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
        )

    def __repr__(self):
        betting_team = f"{self.names[self.betting_team]}/ {self.names[self.betting_team+2%4]}"
        contract = f'team {betting_team} joue à {self.bet_value}{["♠", "♥", "♦", "♣"][self.trump]}'
        hands = [
            f"{name}: {' '.join(get_card_name(card) for card in hand)}" for name, hand in zip(self.names, self.hands)
        ]
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

    hand = gs.hands[(gs.current_lead + len(gs.current_trick)) % 4]

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
    ranks = [0] * 32
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

    def bid(self, game_state: GameState, passed_bid: list[int]):
        return random.choice(game_state.get_legal_bids())


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


def play_one_game(agents: list[Agent], game_state: GameState, verbose=False):
    "update game_state object in place"
    assert len(agents) == 4
    game_state if game_state is not None else GameState.fresh_game()
    scores = get_scores(game_state.trump)

    # if verbose:
    #     print(f'team {game_state.betting_team} prend à {game_state.bet_value}{["♠", "♥", "♦", "♣"][game_state.trump]}')
    #     print("starting hands")
    #     for idx, hand in enumerate(game_state.hands):
    #         print(agents[idx].name, [get_card_name(card) for card in hand])
    #     print("-" * 30)

    while len(game_state.tricks) < 8:
        while len(game_state.current_trick) < 4:
            idx = (game_state.current_lead + len(game_state.current_trick)) % 4
            card = agents[idx].play(game_state.copy())
            assert card in game_state.hands[idx], f"{agents[idx].name} tried to play {get_card_name(card)}"
            game_state.current_trick.append(card)
            game_state.hands[idx].remove(card)
            game_state.last_card_played = card

            if verbose:
                print(
                    f"{agents[idx].name} played {get_card_name(card)}. Current trick is {get_cards_names(game_state.current_trick)}"
                )

        idx += 1
        game_state.leads.append(idx % 4)
        ranks = get_ranks(game_state.trump, game_state.current_trick[0] // 8)
        winner_card = max(game_state.current_trick, key=lambda x: ranks[x])
        indexed_trick = {card: idx % 4 for card, idx in zip(game_state.current_trick, range(idx, idx + 4))}
        ordered_current_trick = list(sorted(game_state.current_trick, key=indexed_trick.get))

        game_state.scores[indexed_trick[winner_card] % 2] += sum(scores[card] for card in game_state.current_trick)

        game_state.current_lead = indexed_trick[winner_card]
        game_state.tricks.append(ordered_current_trick)
        game_state.current_trick = []

        if verbose:
            # print("-" * 30)
            # print(agents[idx % 4].name, "started", [get_card_name(card) for card in game_state.tricks[-1]])
            print(game_state.scores)
            print("-" * 30)

    final_points(game_state)


def play_one_card(card, game_state: GameState, verbose=False):
    "update game_state object in place"
    assert len(game_state.current_trick) < 4 and len(game_state.tricks) < 8

    idx = (game_state.current_lead + len(game_state.current_trick)) % 4
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


def ucb1(parent, child, parent_visit_log, temp=2):
    exploitation = child.value / child.visits
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
        return f"Node(visits={self.visits}, relative_value={self.value/self.visits*100 if self.visits else 0:.2f})"

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
        self.player_index = (game_state.current_lead + len(game_state.current_trick)) % 4

        legal_actions = get_legal_actions(game_state)

        if len(legal_actions) == 1:
            if self.verbose:
                print(f"{self.name}: only legal card {get_card_name(legal_actions[0])} was played")
            return legal_actions[0]

        root = Node(game_state.copy(), parent=None)

        for _ in range(self.iterations):
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

    def traverse(self, node):
        while not node.is_terminal():
            if not node.is_fully_expanded:
                return self.expand(node)
            else:
                node = node.select()
        return node

    def expand(self, node):
        assert (
            sum(1 for hand in node.game_state.hands for _ in hand)
            + sum(1 for trick in node.game_state.tricks for _ in trick)
            + len(node.game_state.current_trick)
            == 32
        ), "Some cards probably fell on the floor..."

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
        raise Exception("Should not reach here")

    def rollout(self, game_state: GameState):
        while len(game_state.tricks) < 8:
            possible_actions = get_legal_actions(game_state)
            action = random.choice(possible_actions)
            play_one_card(action, game_state)
        return game_state.scores[self.player_index % 2] / (game_state.bet_value * game_state.coinche * 2 + 20)

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
        assert last_card_played in get_legal_actions(
            node.game_state
        ), f"{self.name} tried to play {get_card_name(last_card_played)} which is not in their legal cards"

        return last_card_played


class DuckAgent(OracleAgent):
    def __init__(self, name="Duck", iterations=10_000, verbose=False):
        super().__init__(name=name, iterations=iterations, verbose=verbose)

    def play(self, game_state: GameState):

        # for the score index & determinization
        self.player_index = (game_state.current_lead + len(game_state.current_trick)) % 4

        legal_actions = get_legal_actions(game_state)

        if len(legal_actions) == 1:
            if self.verbose:
                print(f"{self.name}: only legal card {get_card_name(legal_actions[0])} was played")
            return legal_actions[0]

        root = Node(None, parent=None)

        for _ in range(self.iterations):

            predicted_game_state = game_state.determinize(self.player_index)  # determinize
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


def main():

    # 46 - 110 carreau - team 2
    # 23 - 90 coeur - team 1

    n_iter = 100_000

    for i in range(1):

        # randys = [
        #     RandomAgent(name="Jean"),
        #     RandomAgent(name="Ivan"),
        #     RandomAgent(name="Jules"),
        #     RandomAgent(name="Eloi"),
        # ]

        # oracles = [
        #     OracleAgent(name="Jean", iterations=n_iter, verbose=False),
        #     OracleAgent(name="Ivan", iterations=n_iter, verbose=False),
        #     OracleAgent(name="Jule", iterations=n_iter, verbose=False),
        #     OracleAgent(name="Eloi", iterations=n_iter, verbose=False),
        # ]

        ducks = [
            DuckAgent(name="Jean", iterations=n_iter, verbose=False),
            DuckAgent(name="Ivan", iterations=n_iter, verbose=False),
            DuckAgent(name="Jule", iterations=n_iter, verbose=False),
            DuckAgent(name="Eloi", iterations=n_iter, verbose=False),
        ]

        game_state = GameState.fresh_game(
            names=["Jean", "Ivan", "Jule", "Eloi"], bet_value=90, betting_team=0, trump=2, coinche=1, seed=23
        )
        # game_state2 = GameState.fresh_game(
        #     names=["Ivan", "Jean", "Eloi", "Jules"], bet_value=110, betting_team=1, trump=2, coinche=1, seed=46
        # )

        print(game_state)
        play_one_game(ducks, game_state, verbose=True)

        # for _ in range(1):
        #     for i in range(4):
        #         card = AGENTS[i].play(game_state)
        #         play_one_card(card, game_state, verbose=True)
        #     print(game_state.scores)
        #     print("-" * 30)

        # for i in range(2):
        #     card = AGENTS[i].play(game_state)
        #     play_one_card(card, game_state, verbose=True)

        # print(game_state)
        # deter = game_state.determinize(0)
        # print([[get_card_name(card) for card in hand] for hand in deter.hands])

        print(game_state)
        print(game_state.scores)


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()

    main()  # Call the main function where your code is executed

    pr.disable()

    pr.dump_stats("profile_results.prof")

    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
