from typing import Dict, List, Tuple
import regex as re
import functools

# Simple string tokeniser

class BPETokeniser():
    def __init__(self) -> None:
        self.vocab = {}
        self.merges = {}
        self.token_lut = {}
        self.pattern = r"\p{L}+|\d{1,3}|\p{P}{1,4}|\s{1,4}"

    @functools.cache
    def get_bytes_for_id(self, id):
        if id not in self.merges:
            raise ValueError(f"Invalid id {id}") 
        
        m = self.merges[id]
        if type(m) is bytes:
            return m
        else:
            return self.get_bytes_for_id(m[0]) + self.get_bytes_for_id(m[1])
        
    def merge(self, content, pair, idx):
        out = []
        
        i = 0
        while i < len(content):
            if i == len(content) - 1:
                out.append(content[i])
                i+=1

            # have to do this way so we can compare against both two byte byte sequences and integer tuples
            elif content[i] == pair[0] and content[1+i] == pair[1]:
                out.append(idx)
                i+=2
            else:
                out.append(content[i])
                i+=1

        return out

    def get_freqs(self, ids):
        freqs: Dict[Tuple[int, int], int] = {}
        for pair in zip(ids, ids[1:]):
            freqs[pair] = freqs.get(pair, 0) + 1
        return freqs

    def train(self, content: str, n_merges: int = 30):
        """
        data: iterable of strings
        """
        
        # Initial vocab to all UTF-8 codepoint bytes
        # We use bytes instead of characters because we want merges to possibly merge unseen unicode too
        # even if more inefficient, better than no compression?
        vocab_size = 256
        vocab = {bytes([i]): i for i in range(256)}     # bytes on int array converts each int (<128) to a byte. Our vocab only deals in bytes

        content = bytearray(content, encoding="utf-8")

        for _ in range(n_merges):
            merge_pair = sorted([(freq, pair) for pair, freq in self.get_freqs(content).items()])[-1]
            if merge_pair[0] == 1:
                break   # If not pairs occur more than once we cannot compress any more
            else:
                merge_pair = merge_pair[1]

            # Dont have to check against reinsertion because it can never happen
            key = vocab[merge_pair] = vocab_size
            vocab_size += 1

            content = self.merge(content, merge_pair, key)
        
        self.vocab = vocab
        self.merges = {id: pair for pair, id in self.vocab.items()}
        self.token_lut = {id: self.get_bytes_for_id(id) for id in self.merges.keys()}
    
    def tokenise(self, content: str):
        content = bytearray(content, encoding = "utf-8")
        
        if len(content) <= 1:
            return content
        
        i = 0
        for pair, idx in self.vocab.items():
            if i < 256:
                i += 1
                continue
            content = self.merge(content, pair, idx)

        return content
    
    def encode(self, content: str):
        out = []
        for group in re.findall(self.pattern, content):
            out.extend(self.tokenise(group))

        return out
    
    def decode(self, tokens):
        out = b''
        for t in tokens:
            out += self.token_lut[t]
        
        return out.decode("utf-8", errors="replace")
        

content = open("test.txt", "r").read()
tk = BPETokeniser()
tk.train(content)
test = "This is an example sentence that I am testig this tokeniser on"
print(len(test))
print(len(tk.encode(test)))
print(tk.decode(tk.encode(test)))
print(tk.decode(tk.encode(test)) == test)
