from tokeniser import BPETokeniser

tokeniser = BPETokeniser()
tokeniser.load("tokeniser_vocab.pkl")

while(True):
    in_seq = input("Enter some text: ")
    tokens = tokeniser.encode(in_seq)
    for tok in tokens:
        print(tokeniser.decode([tok]) + "|", end="")
    print("\n")
    
