from nltk.tokenize import word_tokenize
from lxml import etree
from collections import Counter
from nltk.corpus import stopwords
from string import punctuation
from nltk import WordNetLemmatizer, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

XML_FILE = "news.xml"


class TextProcessing:
    xml: etree
    stop_words: stopwords
    tfidf_matrix: None
    vocabulary: np.asarray

    def __init__(self):
        self.stop_words = stopwords.words("english")
        self.xml = self.read_xml()
        self.tfidf_matrix = None
        self.vocabulary = None

    @staticmethod
    def read_xml():
        return etree.parse(XML_FILE).getroot()

    @staticmethod
    def tokenize_word(news: str):
        return word_tokenize(news.lower())

    @staticmethod
    def lemma_news(tokens: list):
        return list(map(lambda x: WordNetLemmatizer().lemmatize(x), tokens))

    def pun_news(self, lemmas: list):
        return list(filter(lambda x: x not in self.stop_words and x not in punctuation, lemmas))

    @staticmethod
    def postag_news(puns: list):
        return list(map(lambda x: pos_tag([x]), puns))

    @staticmethod
    def noun_tag(postags: list):
        return list(filter(lambda x: x[0][1] == "NN", postags))

    @staticmethod
    def tag_name_part(noun_tags: list):
        return list(map(lambda x: x[0][0], noun_tags))

    @staticmethod
    def calc_tfidf(self):
        news = [" ".join(news) for news in self.token_news]
        vectorizer = TfidfVectorizer()
        self.tfidf_matrix = vectorizer.fit_transform(news).toarray()
        self.vocabulary = np.asarray(vectorizer.get_feature_names())

    @staticmethod
    def reverse_sort_news(name_tags: list):
        """"""
        return sorted(Counter(name_tags).most_common(10), key=lambda x: (x[1], x[0]), reverse=True)[:5]


class KeyTermExtractions(TextProcessing):
    news_dict: dict
    token_news: list

    def __init__(self):
        super().__init__()
        self.news_dict = {}
        self.token_news = []

    def menu(self):
        self.find_words()
        self.print_most_common()

    def find_words(self):
        for news in self.xml[0]:
            token_news = self.tokenize_word(news[1].text.lower())
            lem_news = self.lemma_news(token_news)
            pun_news = self.pun_news(lem_news)
            pos_news = self.postag_news(pun_news)
            tag_news = self.noun_tag(pos_news)
            noun_news = self.tag_name_part(tag_news)
            self.token_news.append(noun_news)

        TextProcessing.calc_tfidf(self)
        for n, news in enumerate(self.xml[0]):
            self.news_dict[news[0].text] = list(zip(self.vocabulary[(-self.tfidf_matrix[n]).argsort()[:10]].tolist(), self.tfidf_matrix[n][(-self.tfidf_matrix[n]).argsort()[:10]].tolist()))
            self.news_dict[news[0].text] = sorted(self.news_dict[news[0].text], key=lambda x: (x[1], x[0]), reverse=True)[:5]

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
