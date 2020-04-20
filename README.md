Introduction
==

Etymology of words if known can help us learn a 
language faster and understand it better. 
To this end I wanted as a thought experiment to train a 
NN to classify words based on their etymology.
Since this is a big undertaking 
_Word Etymologist_ will not fulfill this promise
exactly. For the time being it can be trained
on recognising English words of Greek origin. 
The dataset it is trained on is very limited,
thus evaluation is postponed until more data is 
gathered.
Next we discuss the challenges associated with this problem, 
then the Design Decisions and finally how a classifier can be 
trained.

Issues
==

Large number of roots
--
The different etymological roots 
is so large that is hard to enumerate them entirely.
It is also likely that for many roots we can
find very few words that will make it hard for any
model to be trained reliably on those roots.
My guess is that given enough data we can identify
a set of the most common roots and then we can train
a classifier to identify only those.
Another interesting question to this end is
whether Zipf's law applies in this setting,
since it has proven to describe well data
frequencies in similar applications.


Words with multiple meanings
--
There are a lot of words with multiple
meanings and likely different etymologies,
but not necessarily.
An example is `trunk` which has 4 meanings but
only 2 etymologies according to etymonline.

Contested/Unknown etymologies
--
A lot of etymologies are based on assumptions
and are hard to verify by a non specialist
and often there is no single widely 
accepted etymology.

Invented words
--
Words are invented all the time and there is no 
easy way to track their origin.

Design Decisions
==
I chose to create an English word dataset 
with each word being annotated based on whether 
its origin is Greek or not.
This simplifies classification since it is binary
and I can verify etymology claims I look up 
due to my knowledge of Greek (modern and ancient).
Since it is possible that a word is combined from
a greek and non greek word I annotate each letter
separately, while for words that their Greek origin
is contested I have an intermediate classification
(0 non Greek, 1 contested Greek, 2 Greek).
An example of a word I consider contested is `air`
just because it is also considered a Latin root
and there are words with the prefix
`aero-` in English which is "more" Greek.

It is also unlikely that a word has two meanings
one with a greek root and another one without.
The only example I came up with is `pan`
which is not a word in Greek but rather a prefix.


Training picks random words from dataset to 
train an LSTM Neural Network.
Currently the classifier treats any non 0 annotation
as Greek.
Input takes a small window of word characters 
and classifies the first character in the window.
Words are padded with spaces to classify every
character in the word.
Context matters so the window size 
`W` should be large enough such that it
is possible to classify the first letter of a word
by looking at the first `W` characters. 

LSTM Networks are particularly successful
at POS tagging and this problem is similar in
nature as we don't know the number of letters
in a word and we want to tag whether each letter
of a word is Greek or not.
Models are saved and optionally loaded
again for training. This is essential
as training dataset grows gradually.

 
It is expected that if features of a Greek word 
are picked up from the Network, then even made up words 
that combine existing Greek words will be successfully 
classified.


Run Training
==
To train a NN run

    python train.py --program 1
    
To evaluate a NN run

    python train.py --program 2
    
For more options run

    python train.py --help

    
 
