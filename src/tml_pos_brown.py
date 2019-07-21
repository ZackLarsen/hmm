

import unittest

class CorpusParserTest(unittest.TestCase):
    def setUp():
        self.stream = "\tSeveral/ap defendants/nns ./.\n"
        self.blank = "\t \n"
    def it_parses_a_line(self):
        cp = CorpusParser()
        null = cp.TagWord(word = "START", tag = "START")
        several = cp.TagWord(word = "Several", tag = "ap")
        defendants = cp.TagWord(word = "defendants", tag = "nns")
        period = cp.TagWord(word = ".", tag =".")

        expectations = [
            [null, several],
            [several, defendants],
            [defendants, period]
        ]
        for (token in cp.parse(self.stream)):
            self.assertEqual(token, expectations.pop(0))

        self.assertEqual(len(expectations), 0)

    def it_doesnt_allow_blank_lines(self):
        cp = CorpusParser()
        for(token in cp.parse(self.blank)):
            raise Exception("Should never happen")


class CorpusParser:
    NULL_CHARACTER = "START"
    STOP = "\n"
    SPLITTER = "/"

    class TagWord:
        def __init__(self, **kwargs):
            setattr(self, 'word', kwargs['word'])
            setattr(self, 'tag', kwargs['tag'])

    def __init__(self):
        self.ngram = 2

    def __iter__(self):
        return self

    def next(self):
        char = self.file.read(1)

        if self.stop_iteration: raise StopIteration

        if not char and self.pos != '' and self.word != '':
            self.ngrams.pop(0)
            self.ngrams.append(TagWord(word = self.word, tag = self.tag))
            self.stop_iteration = True
            return self.ngrams

        if char == "\t" or (self.word == "" && STOP.contains(char)):
            return None
        elif char == SPLITTER:
            self.parse_word = false
        elif STOP.contains(char):
            self.ngrams.pop(0)
            self.ngrams.append(TagWord(word = self.word, tag = self.pos))
            self.word = ''
            self.pos = ''
            self.parse_word = True
            return self.ngrams
        elif self.parse_word:
            self.word += char
        else:
            self.pos += char

    def parse(file):
        self.ngrams = [
            TagWord(NULL_CHARACTER, NULL_CHARACTER),
            TagWord(NULL_CHARACTER, NULL_CHARACTER)
        ]

        self.word = ''
        self.pos = ''
        self.parse_word = True
        self.file = file
        return self




'''
Writing the Part-of-Speech Tagger
At this point we are ready to write our part-of-speech tagger class. 
To do this we will have to take care of the following:
1. Take data from the CorpusParser
2. Store it internally so we can calculate the probabilities of word tag combos
3. Do the same for tag transitions
'''


from collections import defaultdict

class POSTagger:
    def __init__(self, data_io):
        self.corpus_parser = CorpusParser()
        self.data_io = data_io
        self.trained = False

    def train():
        if not self.trained:
            self.tags = set(["Start"])
            self.tag_combos = defaultdict(lambda: 0, {})
            self.tag_frequencies = defaultdict(lambda: 0, {})
            self.word_tag_combos = defaultdict(lambda: 0, {})
        for(io in self.data_io):
            for(line in io.readlines()):
                for(ngram in self.corpus_parser.parse(line)):
                    write(ngram)
        self.trained = True

    def write(ngram):
        if ngram[0].tag == 'START':
            self.tag_frequencies['START'] += 1
            self.word_tag_combos['START/START'] += 1

        self.tags.append(ngram[-1].tag)
        self.tag_frequencies[ngram[-1].tag] += 1
        self.word_tag_combos["/".join([ngram[-1].word, ngram[-1].tag])] += 1
        self.tag_combos["/".join([ngram[0].tag, ngram[-1].tag])] += 1

    def tag_probability(previous_tag, current_tag):
        denom = self.tag_frequencies[previous_tag]

        if denom == 0:
            0.0
        else:
            self.tag_combos["/".join(previous_tag, current_tag)] / float(denom)

    def word_tag_probability(word, tag):
        denom = self.tag_frequencies[tag]
        if denom == 0:
            0.0
        else:
            self.word_tag_combos["/".join(word, tag)] / float(denom)

    def probability_of_word_tag(word_sequence, tag_sequence):
        if len(word_sequence) != len(tag_sequence):
            raise Exception('The word and tags must be the same length!')

        length = len(word_sequence)

        probability = 1.0

        for (i in xrange(1, length)):
            probability *= (
                    tag_probability(tag_sequence[i - 1], tag_sequence[i]) *
                    word_tag_probability(word_sequence[i], tag_sequence[i])
            )

        probability

    def viterbi(sentence):
        parts = re.sub(r"([\.\?!])", r" \1", sentence)
        last_viterbi = {}
        backpointers = ["START"]
        for (part in parts[1:]):
            viterbi = {}
        for (tag in self.tags):
            if tag == 'START':
                next()
            if last_viterbi:
                break
            best_previous = max(
                for ((prev_tag, probability) in last_viterbi.iteritems()):
                    probability * \
                    tag_probability(prev_tag, tag) * \
                    word_tag_probability(part, tag)
            )
            best_tag = best_previous[0]
            probability = last_viterbi[best_tag] * \
                      tag_probability(best_tag, tag) * \
                      word_tag_probability(part, tag)
            if probability > 0:
                viterbi[tag] = probability

        last_viterbi = viterbi

        backpointers << (
            max(v for v in last_viterbi.itervalues()) or
            max(v for v in self.tag_frequencies.itervalues())
        )

    backpointers

    for (tag in self.tags):
        if tag == 'START':
            next()
    else:
        probability = tag_probability('START', tag) * \
                      word_tag_probability(parts[0], tag)
    if probability > 0:
        last_viterbi[tag] = probability
    backpointers.append(
        max(v for v in last_viterbi.itervalues()) or
        max(v for v in self.tag_frequencies.itervalues())
    )


import StringIO

class TestPOSTagger(unittest.TestCase):
    def setUp():
        self.stream = StringIO("A/B C/D C/D A/D A/B ./.")
        self.pos_tagger = POSTagger([StringIO.StringIO(self.stream)])
        self.pos_tagger.train()

    def it_calculates_probability_of_word_and_tag(self):
        self.assertEqual(self.pos_tagger.word_tag_probability("Z", "Z"), 0)

        # A and B happens 2 times, count of b happens twice therefore 100%
        self.assertEqual(self.pos_tagger.word_tag_probability("A", "B"), 1)
        # A and D happens 1 time, count of D happens 3 times so 1/3
        self.assertEqual(self.pos_tagger.word_tag_probability("A", "D"), 1.0 / 3.0)
        # START and START happens 1, time, count of start happens 1 so 1
        self.assertEqual(self.pos_tagger.word_tag_probability("START", "START"), 1)
        self.assertEqual(self.pos_tagger.word_tag_probability(".", "."), 1)

    def it_calculates_probability_of_words_and_tags(self):
        words = ['START', 'A', 'C', 'A', 'A', '.']

        tags = ['START', 'B', 'D', 'D', 'B', '.']
        tagger = self.pos_tagger
        tag_probabilities = reduce((lambda x, y: x * y), [
            tagger.tag_probability("B", "D"),
            tagger.tag_probability("D", "D"),
            tagger.tag_probability("D", "B"),
            tagger.tag_probability("B", ".")
        ])

        word_probabilities = reduce((lambda x, y: x * y), [
            tagger.word_tag_probability("A", "B"),  # 1
            tagger.word_tag_probability("C", "D"),
            tagger.word_tag_probability("A", "D"),
            tagger.word_tag_probability("A", "B"),  # 1
        ])

        expected = word_probabilities * tag_probabilities

        self.assertEqual(tagger.probability_of_word_tag(words, tags), expected)

    def viterbi(self):
        training = "I/PRO want/V to/TO race/V ./. I/PRO like/V cats/N ./."
        sentence = 'I want to race.'
        tagger = self.pos_tagger
        expected = ['START', 'PRO', 'V', 'TO', 'V', '.']
        self.assertEqual(pos_tagger.viterbi(sentence), expected)



class CrossValidationTest(unittest.TestCase):
    def setUp(self):
        self.files = Glob('./data/brown/c***')

    FOLDS = 10

    def cross_validate(self):
        for (i in xrange(0,FOLDS)):
            splits = len(self.files) / FOLDS
            self.validation_indexes = range(i * splits, (i + 1) * splits)
            self.training_indexes = range(0, len(self.files)) - self.validation_indexes
            self.validation_files = (file for idx, file in enumerate(self.files))
            self.training_files = (file for idx, file in enumerate(self.files))

            validate(fold)

    def validate(self, fold):
        pos_tagger = POSTagger.from_filepaths(training_files, true)

        misses = 0
        successes = 0

        for(vf in self.validation_files):
            with open(vf,'rb') as f:
                for(l in f.readlines()):
                    if re.match(r"\A\s+\z/", l):
                        next()
                    else:
                        words = []
                        parts_of_speech = ['START']
                        for (ppp in re.split(r"\s+")):
                            z = re.split(r"\/", ppp)
                            words.append(z[0])
                            parts_of_speech.append(z[1])

                        tag_seq = pos_tagger.viterbi(" ".join(words))
                        for (k,v) in zip(tag_seq, parts_of_speech)):
                            misses += sum((k != v) ? 1 : 0
                        for (k,v) in zip(tag_seq, parts_of_speech)):
                            successes += sum((k == v) ? 1 : 0

        print("Error rate was: " + float(misses) / float(successes + misses))



