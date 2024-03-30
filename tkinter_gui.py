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
        self.root.geometry("800x600")
        self.root.configure(bg="dark green")
        self.main_frame = tk.Frame(self.root, bg="dark green")
        self.main_frame.place(relwidth=1, relheight=1)
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
        card_images = {}
        for i, card in enumerate(card_names):
            path = os.path.join("cards/processed/", card + ".png")
            image = Image.open(path)
            image = image.resize((80, 120), Image.LANCZOS)
            card_images[i] = ImageTk.PhotoImage(image)
        return card_images

    def load_card_back_image(self):
        # Assuming the card back image is named "card_back.png" and located in the same folder as the other cards
        path = os.path.join("cards/processed/", "card_back.png")
        image = Image.open(path)
        image = image.resize((80, 120), Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    def display_cards(self):
        # using place and the main frame
        # cards are displayed at the top of the frame for north, at the bottom for south, on the left for west and on the right for east
        for i, player in enumerate(self.game_state.hands):
            frame = self.main_frame
            overlap = 20  # Adjust the overlap amount as needed
            for j, card in enumerate(player):
                if i == 2:
                    image = self.card_images[card]
                else:
                    image = self.card_back_image
                label = tk.Label(frame, image=image)
                if i == 0:  # North
                    label.place(x=360 + j * (80 - overlap), y=10, anchor="n")
                elif i == 1:  # East
                    label.place(x=760, y=240 + j * (120 - overlap), anchor="e")
                elif i == 2:  # South
                    label.place(x=360 - j * (80 - overlap), y=580, anchor="s")
                else:  # West
                    label.place(x=40, y=240 - j * (120 - overlap), anchor="w")


def main():
    root = tk.Tk()
    app = CoincheGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
