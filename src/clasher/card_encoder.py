from .data import CardDataLoader
class CardEncoder:
    def __init__(self, card_data_loader: CardDataLoader):
        self.card_to_index = {card: idx for idx, card in enumerate(card_data_loader.get_all_card_names())}
        self.index_to_card = {idx: card for idx, card in enumerate(card_data_loader.get_all_card_names())}
    
    def encode(self, card_name):
        return self.card_to_index.get(card_name, -1) # return -1 for unknown cards
    
    def decode(self, index):
        return self.index_to_card.get(index, "Unknown")
    
    def one_hot_encode(self, cardlist: list[str]):
        one_hot = [0] * len(self.card_to_index)
        for card_name in cardlist:
            index = self.encode(card_name)
            if index == -1:
                return None  # or raise an exception for unknown card
            one_hot[index] = 1
        return one_hot