Before using text to train llm model, we need to convert text into some kind of mathmatical objects which can be handle by model. The first preprocess step is to convert words into number and convert number to words, for example given text:
"I love you", we need to convert the string into array of numbers such as [1, 2, 3], then given array of numbers we need to convert them back to given text that is given [1, 2, 3] we need to transform it to "I love you".

Let's see how can we do it. First we need some text as base material, we will downloand text from the link: https://en.wikisource.org/wiki/Fire-Tongue/Chapter_1 by using the following code :

```py
import urllib.request 
url = "https://en.wikisource.org/wiki/Fire-Tongue/Chapter_1"
file_path = "fire-tongue.txt"
urllib.request.urlretrieve(url, file_path)

with open("fire-tongue.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()

print(f"Total number of characters: {len(raw_text)}")
print(raw_text)
```

Run above code then we have a text file saved on the disk with name fire-tongue.txt, this file is acutally the html content for the given page, and from above code we can see the output as :

```py
Total number of characters: 57211
<!DOCTYPE html>
<html class="client-nojs" lang="en" dir="ltr">
<head>
<meta charset="UTF-8">

....

```
As we can see that, many words are combine with special characters such as <, >, !, #, let's seperate word from those special symbols, and count each word and symbol as different units by using following code:

```py
#split text according to given chars
import re
preprocessed = re.split(r'([,.:;?_!=\-\"<>#\{\}\'$\&/()\[\]+]|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])
```
Running above code we get the following result:

```py
['<', '!', 'DOCTYPE', 'html', '>', '<', 'html', 'class', '=', '"', 'client', '-', 'nojs', '"', 'lang', '=', '"', 'en', '"', 'dir', '=', '"', 'ltr', '"', '>', '<', 'head', '>', '<', 'meta']

```
As we can see that, symbols like <, ! are seperate out as their own, all the words or symbols are called as tokens, now we can count the number of unique tokens in the text, and assign different number to represent each token, first let's
remove any repeative token and count how many different tokens in the given text:

```py
#assign number to each token and sort them alphabetically
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
#total unique characters
print(vocab_size)
```
Running above code we get following result:
```py
1999
```
As you can see, even the total number of tokens in the text is 57211, but if we count repeative token as one, there are only 1999 unique tokens in the text. Let's check the first 50 unique tokens:

```py
vocab = {token: integer for integer, token in enumerate(all_words)}
#print the first 50 tokens
for i, item in enumerate(vocab.items()):
  print(item)
  if i >= 50:
    break
```
Running above code we get following result:

```py
('!', 0)
('"', 1)
('#', 2)
('$', 3)
('%', 4)
('%2C', 5)
('&', 6)
("'", 7)
('(', 8)
(')', 9)
('*', 10)
('+', 11)
(',', 12)
('-', 13)
('.', 14)
('/', 15)
('0', 16)

...
```
As we can see that, each token has already pair with one number right? By using the same scheme we can assign different number to different token:

```py
#convert token to id, convert id to token
class SimpleTokenizerV1:
  def __init__(self, vocab):
    self.str_to_int = vocab
    self.int_to_str = {i : s for s,i in vocab.items()}

  def encode(self, text):
    #convert sentence into list of ids
    preprocessed = re.split(r'([,.:;?_!=\-\"<>#\{\}\'$\&/()\[\]+]|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    #map token to id
    ids = [self.str_to_int[s] for s in preprocessed]
    return ids 

  def decode(self, ids):
    #convert id list to words
    text = " ".join([self.int_to_str[i] for i in ids])
    #remove space before given symbols
    text = re.sub(r'\s+([,.?"()\'])', r'\1', text)
    return text
```

Then, let's select one sentence from the text, and try above code to convert a sentence with words into array of number as following:

```py
tokenizer = SimpleTokenizerV1(vocab)
text = """
One summer's evening when the little clock upon his table was rapidly approaching the much-desired hour,
"""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
```
Running above code we get the following result:
```py
[332, 1706, 7, 1567, 870, 1934, 1743, 1179, 687, 1825, 1030, 1722, 1869, 1503, 531, 1743, 1262, 13, 776, 1036, 12]
One summer' s evening when the little clock upon his table was rapidly approaching the much - desired hour,
```

As you can see, different token convert to different number, such as One -> 332, summer' -> 1706 , and given a list of numbers, we can convert them into a sentence of several words. But our scheme has a bug, for example running following code
we will get errors:

```py
#the problem is, tokenizer can't handle word not in the text
text = "Hello, do you like tea?"
print(tokenizer.encode(text))
```

That's because the word "Hello" is never appear in the text, therefore the tokenizer dosen't know convert word "Hello" to which number since there is not mapping for word "Hello". To handle such problem, we can use special token to represent
all foreign words that are not shown in the text, for example for any unseen word, we map them to token "*|unk|*", and we use another special "*|endoftext|*" token for the purpose of indicating the concate of two different text source,
therefore we add the following code:

```py
#handle unseen words 
#add to special tokens, *|unk|* and *|endoftext|*, every unseen word map to token |unk|
#|endoftext| as symbol to seperate different text source
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["*|endoftext|*", "*|unk|*"])
vocab = {token : integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-10:]):
  print(item)
```

Running above code we get the following result:

```py
2001
('z', 1991)
('{', 1992)
('|', 1993)
('}', 1994)
('~ext', 1995)
('—', 1996)
('←', 1997)
('→', 1998)
('*|endoftext|*', 1999)
('*|unk|*', 2000)
```
Now let's update the tokenizer to handle unseen words as following:

```py
#convert token to id, convert id to token
class SimpleTokenizerV2:
  def __init__(self, vocab):
    self.str_to_int = vocab
    self.int_to_str = {i : s for s,i in vocab.items()}

  def encode(self, text):
    #convert sentence into list of ids
    preprocessed = re.split(r'([,.:;?_!=\-\"<>#\{\}\'$\&/()\[\]+]|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(f"encode preprocessed: {preprocessed}")
    #add unk token for unseen word 
    preprocessed = [item if item in self.str_to_int else "*|unk|*" for item in preprocessed]
    #map token to id
    ids = [self.str_to_int[s] for s in preprocessed]
    return ids 

  def decode(self, ids):
    #convert id list to words
    text = " ".join([self.int_to_str[i] for i in ids])
    #remove space before given symbols
    text = re.sub(r'\s+([,.?"()\'])', r'\1', text)
    return text
```
Now let's test the new tokenizer by following code:

```py
tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit rerraces of palace."
#notice space in the front and end
text  = " *|endoftext|* ".join((text1,text2))
        
print(text)
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))
```

Run above code we get the following result:

```py
Hello, do you like tea? *|endoftext|* In the sunlit rerraces of palace.
encode preprocessed: ['Hello', ',', 'do', 'you', 'like', 'tea', '?', '*|endoftext|*', 'In', 'the', 'sunlit', 'rerraces', 'of', 'palace', '.']
[2000, 12, 805, 1988, 1169, 2000, 159, 1999, 276, 1743, 2000, 2000, 1325, 2000, 14]
*|unk|*, do you like *|unk|*? *|endoftext|* In the *|unk|* *|unk|* of *|unk|*.
```

As we can see this time, the unseen word "Hello" map to token of *|unk|*.

There is another complicated tokenizing scheme used by chatGPT which has name "byte pair encoding", originally it is a kind of data compression algorithm, let's look into its process. BPE merge frequent used character pair in multiple
iteration. For example given string:

"low low low low low lower lower newest newest newest newest newest newest widest widest widest"

the word frenquency count is as following:

{
"low": 5,
"lower", 2,
"newest", 6,
"widest",: 3,
}

Now, let's split words into characters, and the set of characters will be the ininitial vocab:
{l, o, w, e, r, n, s, t, i, d}

and each word split into collection of characters as following:

{
"l o w": 5,
"l o w e r " : 2,
"n e w e s t": 6,
"w i d e s t": 3,
}

Now let's look at every neighboring character pairs, for example "l o w" has two neighboring pair "lo", "ow", since "lo" appear in "l o w" and "l o w e r", the former has frequent count as 5, and the later has frequent count as 2, then
pair "lo" has frequent count as 7, by this way, we have following:

{
"l o": 7,
"o w": 7,
"w e": 8,
"e r": 2,
"n e": 6,
"e w": 6,
"e s": 9,
"s t": 9,
"w i": 3,
"i d": 3,
"d e": 3,
}

Now we can see the most frequent pair is "e s" and "s t", then we combine "e s" as one unit name "es", then we add es as one into the vocab:
{l, o, w, e, r, n, s, t, i, d, es}
and the collection of each word as :

{
"l o w": 5,
"l o w e r": 2,
"n e w es t": 6,
"w i d es t": 3,
}

Now this time the most frequent char  pair is "es" and "t", then we combine them into one as "est" and add to vocab:
{l, o, w, e, r, n, s, t, i, d, es, est}
and the word char collection as :

{
"l o w": 5,
"l o w e r": 2,
"n e w est": 6,
"w i d est": 3
}

This time the most frequent char pair is “l o", let's add them to vocab:

{l, o, w, e, r, n, s, t, i, d, es, est, lo}

and the word colloection is :

{
"lo w": 5,
"lo w e r": 2,
"n e w est": 6,
"w i d est": 3
}

It is easy to see now the most frequent pair is "lo" and "w", merge them and add to vocab:
{l, o, w, e, r, n, s, t, i, d, es, est, lo, low}

The word char collection is: 
{
"low": 5,
"low e r", 2,
"n e w est", 6,
"w i d est": 3
}

Iterate with this process until the iteration time reach the preset number or the vocab reach desired side. Let's see how to use code to implement the process:

```py
from collections import defaultdict
#count word frequency and split word as char collection
def get_vocab(data):
  vocab = defaultdict(int)
  for word in data.split():
          vocab[' '.join(list(word))] += 1
  return vocab

vocab = get_vocab("low low low low low lower lower newest newest newest newest newest newest widest widest widest")
print(vocab)
```
Run above code we get following result:

```py
defaultdict(<class 'int'>, {'l o w': 5, 'l o w e r': 2, 'n e w e s t': 6, 'w i d e s t': 3})
```

Then we can count the frequency of neighboring char pairs:

```py
#count the freqency of neighboring char pair
from collections import Counter
def get_stats(vocab):
  pairs = Counter()
  for word, freq in vocab.items():
    symbols = word.split()
    #check neighboring char pair
    for i in range(len(symbols) - 1):
      pairs[symbols[i], symbols[i+1]] += freq

  return pairs

pairs = get_stats(vocab)
print(pairs)
```
Running above code we can have following result:

```py
Counter({('e', 's'): 9, ('s', 't'): 9, ('w', 'e'): 8, ('l', 'o'): 7, ('o', 'w'): 7, ('n', 'e'): 6, ('e', 'w'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'e'): 3, ('e', 'r'): 2})
```

Then we can get the pair with hightest frequency and merge them into one:

```py
#merge most frequent pair as one
def merge_vocab(pair, vocab):
  new_vocab = {}
  neighboring_chars = ' '.join(pair)
  #combine two char as one
  replacement = ''.join(pair) 
  for word in vocab:
    new_word = word.replace(neighboring_chars, replacement)
    new_vocab[new_word] = vocab[word] 

  return new_vocab

most_frequent = max(pairs, key=pairs.get)
new_vocab = merge_vocab(most_frequent, vocab)
print(new_vocab)
```

Running above code we get following result:

```py
{'l o w': 5, 'l o w e r': 2, 'n e w es t': 6, 'w i d es t': 3}
```

Let's combine all above steps together and iterate the process for given times:

```py


#let's combine all steps and iterate for given times
def byte_pair_encoding(vocab, num_merges):
  for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
      break
    most_frequent = max(pairs, key=pairs.get)
    vocab = merge_vocab(most_frequent, vocab)
    print(f"merget: {i+1}, most frequent: {most_frequent}")
  return vocab

result_vocab = byte_pair_encoding(vocab, 20)
print("final vocab")
for word in result_vocab:
  print(f"{word}: {result_vocab[word]}")
```

Running above code we get following result:

```py
merget: 1, most frequent: ('e', 's')
merget: 2, most frequent: ('es', 't')
merget: 3, most frequent: ('l', 'o')
merget: 4, most frequent: ('lo', 'w')
merget: 5, most frequent: ('n', 'e')
merget: 6, most frequent: ('ne', 'w')
merget: 7, most frequent: ('new', 'est')
merget: 8, most frequent: ('w', 'i')
merget: 9, most frequent: ('wi', 'd')
merget: 10, most frequent: ('wid', 'est')
merget: 11, most frequent: ('low', 'e')
merget: 12, most frequent: ('lowe', 'r')
final vocab
low: 5
lower: 2
newest: 6
widest: 3
```

From the result we can understand that, byte pair encoding actually is a kind of data compression, it removes any word repeation in the data with given word and its repeat times. The example we have here is only for illustration, the complete
implementation is rather complicate, but lukily there is lib already for it, we can install the following lib to use PBE algorithm:

```py
pip install tiktoken
```
Then we can select given tokenizer by following code:

```py
import tiktoken
tokenizer = tiktoken.get_encoding('gpt2')
```

Let's try the encoding alogrithm like following:

```py
text = ("Hello, do you like a cup of chinese tea? *|endoftext|* In the sunlit terraces"
       "of someunknowPlace.")
integers = tokenizer.encode(text, allowed_special={"*|endoftext|*"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)
```

Running above code we have following result:

```py
[15496, 11, 466, 345, 588, 257, 6508, 286, 442, 3762, 8887, 30, 1635, 91, 437, 1659, 5239, 91, 9, 554, 262, 4252, 18250, 8812, 2114, 1659, 617, 2954, 2197, 27271, 13]
Hello, do you like a cup of chinese tea? *|endoftext|* In the sunlit terracesof someunknowPlace.
```
