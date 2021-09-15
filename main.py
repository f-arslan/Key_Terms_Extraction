from nltk.tokenize import word_tokenize
from lxml import etree
from collections import Counter
from nltk.corpus import stopwords
from string import punctuation
from nltk import WordNetLemmatizer, pos_tag

XML_FILE = "news.xml"


class TextProcessing:
    xml: etree
    stop_words: stopwords

    def __init__(self):
        self.stop_words = stopwords.words("english")
        self.xml = self.read_xml()

    @staticmethod
    def read_xml():
        return etree.parse(XML_FILE).getroot()

    @staticmethod
    def tokenize_word(news: str):
        """Tokenize words with word_tokenize."""
        return word_tokenize(news.lower())

    @staticmethod
    def lemma_news(tokens: list):
        """Return lemmatize word list."""
        return list(map(lambda x: WordNetLemmatizer().lemmatize(x), tokens))

    def pun_news(self, lemmas: list):
        """Return the word list which is not including stop_words and punctuation."""
        return list(
            filter(lambda x: x not in self.stop_words and x not in punctuation, lemmas)
        )

    @staticmethod
    def postag_news(puns: list):
        """Return the word with postag."""
        return list(map(lambda x: pos_tag([x]), puns))

    @staticmethod
    def noun_tag(postags: list):
        """Return the tags which is equal to NOUN("NN")"""
        return list(filter(lambda x: x[0][1] == "NN", postags))

    @staticmethod
    def tag_name_part(noun_tags: list):
        """Get word part from tag_word list."""
        return list(map(lambda x: x[0][0], noun_tags))

    @staticmethod
    def reverse_sort_news(name_tags: list):
        """
        Find the most common 10 words and range in 5, sort by value after that name.
        Minimum number of most_common is 10, you can increase for safety. But run speed will slow.
        """
        return sorted(
            Counter(name_tags).most_common(10), key=lambda x: (x[1], x[0]), reverse=True
        )[:5]


class KeyTermExtractions(TextProcessing):
    news_dict: dict

    def __init__(self):
        super().__init__()
        self.news_dict = {}

    def menu(self):
        self.find_words()
        self.print_most_common()

    def find_words(self):
        """Applying text processing when traveling on the tree."""
        for news in self.xml[0]:
            token_news = self.tokenize_word(news[1].text)
            lem_news = self.lemma_news(token_news)
            pun_news = self.pun_news(lem_news)
            pos_news = self.postag_news(pun_news)
            tag_news = self.noun_tag(pos_news)
            noun_news = self.tag_name_part(tag_news)
            self.news_dict[news[0].text] = self.reverse_sort_news(noun_news)

    def print_most_common(self):
        for key, value in self.news_dict.items():
            print(key + ":")
            for word in value:
                print(word[0], end=" ")
            print("\n")


def main():
    key = KeyTermExtractions()
    key.menu()


if __name__ == "__main__":
    main()
