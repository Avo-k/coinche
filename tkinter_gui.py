import tkinter as tk
from PIL import Image, ImageTk
import os
import random
from game_state import GameState

SUITS = "♠♥♦♣"
VALUES = ["7", "8", "9", "10", "J", "Q", "K", "A"]


class CoincheGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Coinche")
        self.root.geometry("900x500")
        self.root.configure(bg="dark green")

        self.main_frame = tk.Frame(self.root, bg="dark green")
        self.main_frame.grid(sticky="nsew")  # Use grid instead of pack

        self.game_state = GameState.fresh_game(
            names=["North", "East", "South", "West"],
            do_bidding_phase=False,
            verbose=True,
        )

        self.card_images = self.load_card_images()
        self.card_back_image = self.load_card_back_image()
        self.display_cards()

    def load_card_images(self):
        card_names = [f"{value}{suit}" for suit in SUITS for value in VALUES]
        random.shuffle(card_names)  # Shuffle the cards
        card_images = {}
        for i, card in enumerate(card_names):
            path = os.path.join("cards/processed/", card + ".png")
            image = Image.open(path)
            image = image.resize((100, 150), Image.LANCZOS)
            card_images[i] = ImageTk.PhotoImage(image)
        return card_images

    def load_card_back_image(self):
        # Assuming the card back image is named "card_back.png" and located in the same folder as the other cards
        path = os.path.join("cards/processed/", "card_back.png")
        image = Image.open(path)
        image = image.resize((100, 150), Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    def display_cards(self):
        # using grid and the players frames
        # cards are displayed at the top of the frame for north, at the bottom for south, on the left for west and on the right for east
        for i, player in enumerate(self.game_state.hands):
            # frame = self.player_frames[i]
            frame = self.main_frame
            for j, card in enumerate(player):
                if i == 2:
                    image = self.card_images[card]
                else:
                    image = self.card_back_image
                label = tk.Label(frame, image=image)
                starting_row = [0, 1, 9, 1]
                starting_column = [1, 9, 1, 0]

                if i % 2:  # vertical
                    label.grid(row=starting_row[i] + j, column=starting_column[i])
                else:  # horizontal
                    label.grid(row=starting_row[i], column=starting_column[i] + j)


def main():
    root = tk.Tk()
    app = CoincheGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
