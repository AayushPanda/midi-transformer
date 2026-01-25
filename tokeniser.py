import json
import os
from array import array
import heapq
import pickle
from typing import Dict, List, Optional, Tuple
import regex as re
import functools

# Simple string tokeniser

class BPETokeniser():
    def __init__(self) -> None:
        self.vocab_size = 0
        self.vocab = {}
        self.merges = {}
        self.token_lut = {}
        self.pattern = re.compile(r"\p{L}+|\d{1,3}|\p{P}{1,4}|\s{1,4}")

    @functools.cache
    def get_bytes_for_id(self, id):
        if id not in self.merges:
            raise ValueError(f"Invalid id {id}") 
        
        m = self.merges[id]
        if type(m) is bytes:
            return m
        else:
            return self.get_bytes_for_id(m[0]) + self.get_bytes_for_id(m[1])
        
    def merge(self, content, pair, idx, freqs: Optional[List[Tuple[int, Tuple[int, int]]]] = None) -> Tuple[array[int], List[Tuple[int, Tuple[int, int]]]]:
        out = array(content.typecode)
        newfreqs = {}
        i = 0
        while i < len(content):
            if i == len(content) - 1:
                out.append(content[i])
                i+=1

            # have to do this way so we can compare against both two byte byte sequences and integer tuples
            elif content[i] == pair[0] and content[1+i] == pair[1]:
                if freqs:
                    if i > 0:
                        newfreqs[(content[i-1], idx)] = newfreqs.get((content[i-1], idx), 0) + 1
                    if i < len(content) - 1:
                        newfreqs[(idx, content[i+1])] = newfreqs.get((idx, content[i+1]), 0) + 1
                out.append(idx)
                i+=2
            else:
                out.append(content[i])
                i+=1
        if freqs:
            for pair, count in newfreqs.items():
                heapq.heappush(freqs, (-count, pair))
            return out, freqs
        else:
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
        vocab: Dict[Tuple[int, int] | bytes, int] = {bytes([i]): i for i in range(256)}     # bytes on int array converts each int (<128) to a byte. Our vocab only deals in bytes

        content: array[int] = array('H',array('B', bytes(content, encoding="utf-8"))) # type: ignore

        freqs: List[Tuple[int, Tuple[int, int]]] = [(-count, pair) for pair, count in self.get_freqs(content).items()]
        heapq.heapify(freqs)
        for _ in range(n_merges):
            freq, merge_pair = heapq.heappop(freqs)
            freq = -freq
            if freq == 1:
                break   # If not pairs occur more than once we cannot compress any more
            
            # Dont have to check against reinsertion because it can never happen
            key = vocab[merge_pair] = vocab_size
            vocab_size += 1

            content, freqs = self.merge(content, merge_pair, key, freqs)
        
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.merges = {id: pair for pair, id in self.vocab.items()}
        self.token_lut = {id: self.get_bytes_for_id(id) for id in self.merges.keys()}
    
    def tokenise(self, content: str):
        content = array('H', array('B', bytes(content, encoding = "utf-8")))

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
        out = bytearray()
        for t in tokens:
            out.extend(self.token_lut[t])
        return out.decode("utf-8", errors="replace")
    
    def save(self, path: str):
        with open(path, "wb+") as f:
            pickle.dump(self.vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.vocab = pickle.load(f)
            self.vocab_size = len(self.vocab.keys())
            self.merges = {id: pair for pair, id in self.vocab.items()}
            self.token_lut = {id: self.get_bytes_for_id(id) for id in self.merges.keys()}

if __name__ == "__main__":
            
    import time
    import sys

    content = open("test.txt", "r").read()
    n_bytes = len(bytearray(content, encoding="utf-8"))
    test = "This is an example sentence that I am testig this tokeniser on"

    tk = BPETokeniser()

    # ---- Training performance ----
    t0 = time.perf_counter()
    tk.train(content, n_merges=8000)
    t1 = time.perf_counter()

    # ---- Encoding performance ----
    t2 = time.perf_counter()
    encoded = tk.encode(test)
    t3 = time.perf_counter()

    # ---- Decoding performance ----
    t4 = time.perf_counter()
    decoded = tk.decode(encoded)
    t5 = time.perf_counter()

    # ---- Statistics ----
    orig_bytes = len(test.encode("utf-8"))
    token_count = len(encoded)
    decoded_bytes = len(decoded.encode("utf-8"))

    avg_bytes_per_token = (
        sum(len(tk.token_lut[t]) for t in encoded) / token_count
        if token_count > 0 else 0
    )

    print("\n=== BPE Tokeniser Performance Stats ===")
    print(f"Training time       : {(t1 - t0):.2f} s")
    print(f"Encoding time       : {(t3 - t2)*1000:.4f} ms")
    print(f"Decoding time       : {(t5 - t4)*1000:.4f} ms")
    print(f"N training bytes    : {n_bytes} bytes")
    print(f"Training byte/time  : {n_bytes/(t1 - t0):.2f} bps")
    print(f"Encoding time/byte  : {orig_bytes/(t3 - t2):.4f} bps ")
    print(f"Decoding time/token : {token_count/((t5 - t4)):.4f} tps")

    print("\n--- Token statistics ---")
    print(f"Original bytes      : {orig_bytes}")
    print(f"Token count         : {token_count}")
    print(f"Compression ratio   : {orig_bytes / token_count:.2f} bytes/token")
    print(f"Avg bytes/token     : {avg_bytes_per_token:.2f}")

    print("\n--- Vocabulary ---")
    print(f"Vocab size          : {len(tk.vocab)}")
    print(f"Merge rules         : {len(tk.vocab) - 256}")

    print("\n--- Correctness ---")
    print(f"Round-trip correct  : {decoded == test}")

    tk.save("tokeniser_vocab.pkl")
