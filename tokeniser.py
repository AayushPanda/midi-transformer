import json
import os
from array import array
import heapq
import pickle
from typing import Dict, Iterable, List, Optional, Set, Tuple
import regex as re
import functools
from collections import deque


class Node:
    """Represents a single node in a linked list."""
    def __init__(self, data, prev=None, next=None):
        self.data = data
        self.prev: Optional[Node] = prev
        self.next: Optional[Node] = next
    def __str__(self) -> str:
        return str(self.data)
    def __repr__(self) -> str:
        return self.__str__()
    def __lt__(self, other: "Node") -> bool:
        return self.data < other.data

class LinkedList:
    """Represents the linked list structure."""
    def __init__(self, head = None):
        self.head = head

    @classmethod
    def from_list(cls, l: Iterable) -> "LinkedList":
        if not l: return LinkedList()
        head = None
        prev = None
        for item in l:
            curr = Node(item,prev)
            if not head: head = curr
            if prev: prev.next = curr
            prev = curr
        return LinkedList(head)
    
    def __iter__(self):
        curr = self.head
        while(curr):
            yield curr.data
            curr = curr.next

    def __str__(self) -> str:
        output = []
        curr = self.head
        while(curr):
            output.append(curr.data)
            curr = curr.next

        return str(output)
    def __repr__(self) -> str:
        return self.__str__()
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
    
    def get_pair_indices(self, content, ll: LinkedList):
        pairs = {}
        curr = ll.head
        while(curr and curr.next):
            target = (content[curr.data], content[curr.next.data])
            if target not in pairs: pairs[target] = set()
            pairs[target].add(curr)
            curr = curr.next
        return pairs

    def merge(self, content, pair, idx, pairs: Dict[Tuple[int, int], Set[Node]], freqs: Optional[List[Tuple[int, Tuple[int, int]]]] = None) -> Tuple[array[int], List[Tuple[int, Tuple[int, int]]],  Dict[Tuple[int, int], Set[Node]]]:
        if pair not in pairs:
            output = [content]
            if freqs: output.append(freqs)
            output.append(pairs)
            return tuple(output)

        newfreqs = {}
        last_merge_idx = 0
        l = list(pairs[pair])
        for node in sorted(l):
            if not node.next: break
            if node.data == -1: continue    # already merged into something, does not exist anymore
            # if (content[node.data], content[node.next.data]) != pair: continue    # if something stops working for overlapping merges im giving up and using this
            if node.prev:
                target = (content[node.prev.data], idx)
                if freqs:
                    newfreqs[target] = newfreqs.get(target, 0) + 1

                if target not in pairs:
                    pairs[target] = set()
                pairs[target].add(node.prev)

                pairs[(content[node.prev.data], content[node.data])].remove(node.prev)
                if not pairs[(content[node.prev.data], content[node.data])]: del pairs[(content[node.prev.data], content[node.data])]
            if node.next.next:
                target = (idx, content[node.next.next.data])
                if freqs:
                    newfreqs[target] = newfreqs.get(target, 0) + 1


                pairs[(content[node.next.data], content[node.next.next.data])].remove(node.next)
                
                if target not in pairs:
                    pairs[target] = set()

                pairs[target].add(node)
                
                if not pairs[(content[node.next.data], content[node.next.next.data])]: del pairs[(content[node.next.data], content[node.next.next.data])]

            # content[node.next.data] = 6969  # for debugging really
            node.next.data = -1 # mark as consumed
            node.next = node.next.next
            if node.next: node.next.prev = node
            content[node.data] = idx
        
        if pair in pairs: del pairs[pair]
        output = [content]
        if freqs:
            for pair, count in newfreqs.items():
                heapq.heappush(freqs, (-count, pair))
            output.append(freqs)
        output.append(pairs)
        return tuple(output)   # in order, [content, freqs, pairs]

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
        pair_ll = LinkedList.from_list(range(len(content)))
        heapq.heapify(freqs)
        pairs = self.get_pair_indices(content, pair_ll)
        for _ in range(n_merges):
            freq, merge_pair = heapq.heappop(freqs)
            while merge_pair not in pairs:
                freq, merge_pair = heapq.heappop(freqs)
            freq = -freq
            if freq == 1:
                break   # If not pairs occur more than once we cannot compress any more
            
            # Dont have to check against reinsertion because it can never happen
            key = vocab[merge_pair] = vocab_size
            vocab_size += 1
            content, freqs, pairs = self.merge(content, merge_pair, key, pairs, freqs)
        
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.merges = {id: pair for pair, id in self.vocab.items()}
        self.token_lut = {id: self.get_bytes_for_id(id) for id in self.merges.keys()}
    
    def tokenise(self, content: str):
        orig = content
        if not content: return []
        content = array('H', array('B', bytes(content, encoding = "utf-8")))

        if len(content) <= 1:
            return content
        
        i = 0
        ll = LinkedList.from_list(range(len(content)))
        pairs = self.get_pair_indices(content, ll)
        for pair, idx in self.vocab.items():
            if i < 256:
                i += 1
                continue
            content, pairs = self.merge(content, pair, idx, pairs)

        output = []
        curr = ll.head
        while(curr):
            if curr.data != -1:
                output.append(content[curr.data])
            curr = curr.next
        return output
    
    def encode(self, content: str):
        out = []
        for group in re.findall(self.pattern, content):
            out.extend(self.tokenise(group))
        # out = self.tokenise(content)
        return out
    
    def decode(self, tokens, show_boundaries=False):
        out = bytearray()
        token_boundaries = deque()
        for t in tokens:
            out.extend(self.token_lut[t])
            token_boundaries.append(len(self.token_lut[t]))

        output = out.decode("utf-8", errors="replace")

        if not show_boundaries: return output
        
        prefix = []
        run_len = 0
        for c in output:
            prefix.append(c)
            run_len += len(c.encode("utf-8"))
            if run_len >= token_boundaries[0]:
                prefix.append("|")
                run_len = 0
                token_boundaries.popleft()
        return "".join(prefix)

    
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

    content = open("shakespeare.txt", "r").read()
    n_bytes = len(bytearray(content, encoding="utf-8"))
    test = content

    tk = BPETokeniser()

    # ---- Training performance ----
    t0 = time.perf_counter()
    tk.train(content, n_merges=8000)
    # tk.load("tokeniser_vocab.pkl")
    t1 = time.perf_counter()
    print("Done training")

    # ---- Encoding performance ----
    t2 = time.perf_counter()
    encoded = tk.encode(test)
    t3 = time.perf_counter()
    print("Done encoding")

    # ---- Decoding performance ----
    t4 = time.perf_counter()
    decoded = tk.decode(encoded)
    t5 = time.perf_counter()
    print("Done decoding")

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
