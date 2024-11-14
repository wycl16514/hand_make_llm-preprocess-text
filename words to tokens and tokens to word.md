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
