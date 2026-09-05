import nltk
from nltk.corpus import words

# 1. Download the corpus (only needed the first time)
nltk.download('words')

# 2. Define how many tokens you want
num_tokens = 20

# 3. Get the list and slice it to your desired length
my_word_list = words.words()[:num_tokens]

print(my_word_list)