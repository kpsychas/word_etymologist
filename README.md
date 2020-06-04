Introduction
==

This project provides a platform for training a Neural Network
on classification of words based on their etymology. 
The Network can be trained interactively from command line or from a simple GUI
or using a small dataset that distinguishes words between those of 
Greek and non Greek origin.

Next we discuss the motivation the challenges associated with this problem, 
then the Design Decisions and finally how a classifier can be 
trained.

Motivation
==
Etymology of words, if known, can help us learn a 
language faster and understand it better. 
Learning any etymology is a big undertaking so
_Word Etymologist_ partially fulfills this promise.
For the time being it can be trained on recognising 
English words of Greek origin from a limited dataset 
or trained interactively to perform custom classification.

An ideal fully featured solution should consist of a web service
in which the user can do the following.
- Contribute data for classification
- Train different Neural Network models with data contributed by community
or by themselves.
- Use already trained models for prediction.


Challenges
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
meanings and likely different etymologies.
An example is `trunk` which has 4 meanings and
2 etymologies according to 
[etymonline dictionary](https://www.etymonline.com/word/trunk).

Contested/Unknown etymologies
--
A lot of etymologies are based on assumptions
and are hard to verify by a non specialist
and often there is no single widely 
accepted etymology.

Invented words
--
Words are invented all the time and there is no 
easy way to track the origin of newly invented words.


Design Decisions
==

Decisions for automated training
--

I chose to create an English word dataset 
with each word being annotated based on whether 
its origin is Greek or not.
This simplifies classification since it is binary
and I can verify etymology claims I look up 
due to my knowledge of Greek (modern and ancient).
Since it is possible that a word is combined from
a Greek and non Greek word I annotate each letter
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
which actually is a prefix of Greek origin
rather than a word.

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

It is expected that if features of a Greek word 
are picked up from the Network, then even made up words 
that combine existing Greek words will be successfully 
classified.

Decisions for interactive training
--
Interactive training works with Bidirectional LSTMs.
For each different length of words a different network 
needs to be created.
Number of parameters are independent of the length of
a word so parameters are loaded from one model to another
after they are trained on an input. 
Since the training is interactive,
this does not delay training in any significant way.


Other Decisions
--
Models and their parameters are saved to root folder
and with default names that depend on the 
number of hidden layers of the model and window size.

In other words naming is by convention rather than 
configuration, but until custom naming is supported
an alternative is to backup/rename trained files 
of models that you want to train from scratch.

Run Training
==

To automatically train a NN first uncompress `words.csv` file 
from `words.zip` compressed file and then run

    python train.py --program 1
    
To evaluate a previously trained NN run

    python train.py --program 2

To train a NN interactively run

    python train.py --program 3

**[Recommended]**
To train a NN interactively with a GUI run

    python train.py --program 4

For more options run

    python train.py --help
    
Expected to work on Python 3.6 or later 

Contributing
==

Check the `CONTRIBUTING.md` file.


