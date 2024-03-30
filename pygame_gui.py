import itertools as itt
import os
import random

import pygame
from PIL import Image

from game_state import Agent, DuckAgent, GameState, play_one_card, get_empty_info_dict, sort_cards


# Initialize Pygame
pygame.init()

# Set up the game window
WIN_WIDTH = 800
WIN_HEIGHT = 800

CARD_WIDTH, CARD_HEIGHT = 100, 150
window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Belote")


class HumanGuiAgent(Agent):
    def __init__(self, name, player_index):
        super().__init__(name)
        self.player_index = player_index

    def play(self, game_state):
        pass

    def bid(self, game_state):
        pass


def load_images():
    card_names = [f"{value}{suit}" for suit in "♠♥♦♣" for value in ["7", "8", "9", "10", "J", "Q", "K", "A"]]
    card_names.append("card_back")
    card_images = []
    for card in card_names:
        path = os.path.join("cards/processed/", card + ".png")
        image = Image.open(path)
        image = image.resize((CARD_WIDTH, CARD_HEIGHT), Image.LANCZOS)
        card_images.append(pygame.image.fromstring(image.tobytes(), image.size, image.mode))
    return card_images


CARD_IMAGES = load_images()


class Card:
    def __init__(self, image, x, y):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.back = CARD_IMAGES[-1]

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONUP:
            if self.rect.collidepoint(event.pos):
                return True
        return False


CARDS = [Card(CARD_IMAGES[i], 0, 0) for i in range(32)]


# game_state = GameState.fresh_game(
#     names=["North", "East", "South", "West"],
#     do_bidding_phase=False,
#     current_lead=0,  # if change then update players chain before using it
#     verbose=True,
# )


deck = list(range(32))
random.shuffle(deck)
hands = [deck[i * 8 : i * 8 + 8] for i in range(4)]
game_state = GameState(
    names=["North", "East", "South", "West"],
    bids=[],
    bet_value=80,
    betting_team=0,
    trump=0,
    coinche=1,
    hands=hands,
    tricks=[],
    leads=[],
    current_trick=[],
    current_lead=0,
    last_card_played=-1,
    scores=[0, 0],
    info=get_empty_info_dict(),
)


def calculate_card_position(idx, i, hand, margins, vertical_spacing, horizontal_spacing):
    if idx % 2:  # East and West
        x = margins if idx == 3 else WIN_WIDTH - CARD_WIDTH - margins
        total_card_height = (len(hand) - 1) * vertical_spacing + CARD_HEIGHT
        start_y = (WIN_HEIGHT - total_card_height) // 2
        y = start_y + i * vertical_spacing
    else:  # North and South
        x = (WIN_WIDTH - len(hand) * (CARD_WIDTH - horizontal_spacing)) // 2 + i * horizontal_spacing
        y = margins if idx == 0 else WIN_HEIGHT - CARD_HEIGHT - margins
    return x, y


def display_card(window, card, x, y, idx):
    # image = CARD_IMAGES[card]
    # card_object = Card(image, x, y)
    card_object = CARDS[card]
    card_object.rect.topleft = (x, y)
    if idx == 2:
        window.blit(card_object.image, (x, y))
    else:
        window.blit(card_object.back, (x, y))


def display_trick_card(window, card, player_idx):
    x = WIN_WIDTH // 2 - CARD_WIDTH // 2
    y = WIN_HEIGHT // 2 - CARD_HEIGHT // 2

    # Adjust the position based on the player index
    if player_idx == 0:  # North
        y -= 70
    elif player_idx == 1:  # East
        x += 70
    elif player_idx == 2:  # South
        y += 70
    elif player_idx == 3:  # West
        x -= 70

    image = CARD_IMAGES[card]
    card_object = Card(image, x, y)
    window.blit(card_object.image, (x, y))


def display_cards(game_state, window):
    "Display cards slightly overlapping"
    horizontal_spacing = 50
    vertical_spacing = 50
    margins = 20

    # hands
    for idx, hand in enumerate(game_state.hands):
        hand = sort_cards(hand, game_state.trump)
        for i, card in enumerate(hand):
            x, y = calculate_card_position(idx, i, hand, margins, vertical_spacing, horizontal_spacing)
            display_card(window, card, x, y, idx)

    # current_trick
    # display each card of the current trick in the center, in front of their player
    if game_state.current_trick:
        for idx, card in enumerate(game_state.current_trick):
            display_trick_card(window, card, (idx + game_state.current_lead) % 4)
    elif game_state.tricks:
        for idx, card in enumerate(game_state.tricks[-1]):
            display_trick_card(window, card, idx)


n_iter = 100000
thinking_time = 5
players = [
    DuckAgent(name="North", player_index=0, thinking_time=thinking_time, iterations=n_iter, verbose=True),
    DuckAgent(name="East", player_index=1, thinking_time=thinking_time, iterations=n_iter, verbose=True),
    # DuckAgent(name="South", player_index=2, thinking_time=thinking_time, iterations=n_iter, verbose=True),
    HumanGuiAgent(name="South", player_index=2),
    DuckAgent(name="West", player_index=3, thinking_time=thinking_time, iterations=n_iter, verbose=True),
]

next_player = players[game_state.get_current_player()]
print(f"{next_player.name}'s turn")


just_played = True
running = True
while running and len(game_state.tricks) < 8:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONUP:
            if next_player.name == "South":
                mouse_pos = pygame.mouse.get_pos()
                legal_cards = sort_cards(game_state.get_legal_actions(), game_state.trump)
                for card in legal_cards[::-1]:  # sorted than reversed for overlaping
                    card_object = CARDS[card]
                    if card_object.rect.collidepoint(mouse_pos):
                        break
                else:
                    print("was not a valid card")
                    continue
                play_one_card(card, game_state, verbose=True)
                next_player = players[game_state.get_current_player()]
                print(f"{next_player.name}'s turn")
                just_played = True

    if next_player.name != "South" and not just_played:
        card = next_player.play(game_state)
        play_one_card(card, game_state, verbose=True)
        next_player = players[game_state.get_current_player()]
        print(f"{next_player.name}'s turn")

    # Clear the screen
    window.fill((0, 100, 0))  # Green background color

    display_cards(game_state, window)

    pygame.display.flip()

    just_played = False


# Quit the game
pygame.quit()
