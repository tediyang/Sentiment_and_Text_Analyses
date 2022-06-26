import pandas as pd
import urllib.request as ul
from bs4 import BeautifulSoup as soup
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textstat.textstat import textstatistics
import re


class ArticleExtractor(object):
    def __init__(self, file):
        self.file = file

    def file_reader(self):
        # read the file
        articles_links = pd.read_excel(self.file)
        
        return articles_links
    
    def links(self):
        # get the url ids and url links
        ids = [id for id in self.file_reader().URL_ID]
        links = [link for link in self.file_reader().URL]
        
        return ids, links

    def extract(self):
        
        files = []
        # request the url links from the internet
        for id, url in zip(self.links()[0], self.links()[1]):
            
            files.append(f'{id}.txt')
            req = ul.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0')
            req.add_header('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8')
            client = ul.urlopen(req)
            htmldata = client.read()
            client.close()
            
            # paste the data fetched as a html file 
            page = soup(htmldata, "html.parser")
            title = page.find('h1', {'class':"entry-title"}).text
            content = page.find('div', {'class':"td-post-content"})
            

            # get the title and body of the article and save it as a text file using the ids for naming.
            with open(f'{id}.txt', 'w', encoding="utf-8") as output:
                output.write(title)

                # check if there's an author reference in the text, and remove  the reference.
                if content.find('pre', {'class': 'wp-block-preformatted'}):
                    author = content.find('pre', {'class': 'wp-block-preformatted'}).text
                
                for text in content:
                    if author:
                        if text.text != author:
                            output.write(text.text)
                
                    else:
                        output.write(text.text) 

            message = 'Articles extracted succcessfully'  
              
        # returns the message and a list containing the names of the article extracted.
        return message, files


   
link =ArticleExtractor('input.xlsx')
message, files = link.extract()
ids, links = link.links()

print(message)
   
# read the stop words files and add encoding error parameter to make some text files readable by 
# the encoder.
stop_aud = pd.read_table('./StopWords/StopWords_Auditor.txt', header=None)
stop_cur = pd.read_table('./StopWords/StopWords_Currencies.txt', header=None, encoding_errors='replace')
stop_dates = pd.read_table('./StopWords/StopWords_DatesandNumbers.txt', header=None)
stop_gen = pd.read_table('./StopWords/StopWords_Generic.txt', header=None)
stop_genl = pd.read_table('./StopWords/StopWords_GenericLong.txt', header=None)
stop_geo = pd.read_table('./StopWords/StopWords_Geographic.txt', header=None)
stop_name = pd.read_table('./StopWords/StopWords_Names.txt', header=None)


# remove unnecessary characters in the files 
def data_cleaner(data):
    '''since all the stopwords list consist of a single column I'll use the index of the column as 0'''
    name = data.columns[0]
    for n in data[name]:
        if '|' in str(n):
            data.loc[data[name].str.split('|').str.len() == 2, name] = data[name].str.split('|').str[0]
        
        else:
            data.loc[data[name] == n, name] = n

    return data

# get all the stop words as a single file.
def get_stop_words():

    stop_aud_ = data_cleaner(stop_aud)
    stop_cur_ = data_cleaner(stop_cur)
    stop_dates_ = data_cleaner(stop_dates)
    stop_gen_ = data_cleaner(stop_gen)
    stop_genl_ = data_cleaner(stop_genl)
    stop_geo_ = data_cleaner(stop_geo)
    stop_name_ = data_cleaner(stop_name)

    stop_words = pd.concat([stop_aud_, stop_cur_, stop_dates_, stop_gen_, stop_genl_, stop_geo_, stop_name_], axis=0)

    return stop_words


stop_words = get_stop_words()

# read the dictionary file
neg_list = pd.read_table('./MasterDictionary/negative-words.txt', header=None, skiprows=2, encoding_errors= 'replace')
pos_list = pd.read_table('./MasterDictionary/positive-words.txt', header=None, skiprows=1)


# classify the words into negative and positive words.
def get_dictionary(data, data2, stop_words):
    wordp = []
    wordn = []
    for m, n in zip(data[0], data2[0]):
        if m not in stop_words[0].str.lower():
            wordp.append(m)
        
        if n not in stop_words[0].str.lower():
            wordn.append(n)

    dictionary = {'positive_words': wordp, 'negative_words': wordn}
   
    return dictionary

dictionary = get_dictionary(pos_list, neg_list, stop_words)




class TextAnalysis(object):
    def __init__(self, files, dictionary):
        self.files = files
        self.dictionary = dictionary

    #  the sentiment analysis
    def sentiment(self):

        positive_scores = []
        negative_scores = []
        polarity_scores = []
        subjectivity_scores = []

        for file in self.files:
            with open(file, 'r', encoding='utf-8') as f:
                s = f.read()
            
            # separate the words in the file.
            word_tokens = word_tokenize(str(s))
            pos_score = []
            neg_score = []

            # check the classification of each word, if positive add 1 and if negative add -1
            for word in word_tokens:
                if word in self.dictionary['positive_words']:
                    pos_score.append(1)
                
                elif word in self.dictionary['negative_words']:
                    neg_score.append(-1)


            # clean the words by removing the special characters and stop words.
            def clean_word():
                
                stop_words = set(stopwords.words('english'))
                special = '@_!#$%^&*()<>?/\|}{~:;.[]'

                clean_stop = [word for word in word_tokens if word not in stop_words]
                clean_words = [word for word in clean_stop if word not in special]

                return clean_words
            
            clean_words = clean_word()
            total_clean = len(clean_words)

            # calculate the pos, neg, pol, and sub scores for each file and add them to their
            # respective lists.
            positive_score = sum(pos_score)
            negative_score = sum(neg_score) * -1
            polarity_score = (positive_score - negative_score) /\
                             ((positive_score + negative_score) + 0.000001)
            subjectivity_score = (positive_score + negative_score) / ((total_clean) + 0.000001)


            positive_scores.append(positive_score)
            negative_scores.append(negative_score)
            polarity_scores.append(round(polarity_score, 2))
            subjectivity_scores.append(round(subjectivity_score, 2))


        return positive_scores, negative_scores, polarity_scores, subjectivity_scores


    # readability analysis
    def readability(self):

        special = '@_!#$%^&*()<>?/\|}{~:;.[]'
        average_sentence_length = []
        percentage_of_complex_words = []
        fog_index = []


        for file in self.files:
            with open(file, 'r', encoding='utf-8') as f:
                s = f.read()

            # get the sentences from each file and get the sum
            sent_tokens = sent_tokenize(s)
            num_of_words = 0
            num_of_sents = 0
            # for each sentence, get the number of words after only the special characters are removed
            for sent in sent_tokens:
                num_of_words += len([word for word in word_tokenize(sent) if word not in special])
                num_of_sents += 1

            # calculate the average sentence length for each file
            ave_sent_len = num_of_words/num_of_sents
            average_sentence_length.append(round(ave_sent_len, 2))

            # get the number of clean words (special characters and stop words removed.)
            def word_count():

                word_tokens = word_tokenize(s)
                clean_words = [word for word in word_tokens if word not in special]
                words = len([word for word in clean_words])
                
                return words

            # get the number of complex words (syllaables > 2)
            def complex_words():
     
                word_tokens = word_tokenize(s)

                clean_words = [word for word in word_tokens if word not in special]
     
                complex_words = []

                #  using syllable_count from the library textstatistics to get the number of 
                # syllables in each words, and a get the syllables > 2.
                for word in clean_words:
                    syllable_count = textstatistics().syllable_count(word)
                    if syllable_count > 2:
                        complex_words.append(word)
                    
                return len(complex_words)
                        
 
            
            num_of_complex = complex_words()
            num_of_words = word_count()

            # calculate the per of complex words and fog index for each article.
            per_of_complex_words = (num_of_complex / num_of_words) * 100
            percentage_of_complex_words.append(round(per_of_complex_words, 2))

            fog_ind = 0.4 * (ave_sent_len + per_of_complex_words)
            fog_index.append(round(fog_ind, 2))

        return average_sentence_length, percentage_of_complex_words, fog_index


    def average_words_per_sentence(self):

        average_num_of_words_per_sentences = []

        special = '@_!#$%^&*()<>?/\|}{~:;.[]'
        
        for file in self.files:
            with open(file, 'r', encoding='utf-8') as f:
                s = f.read()
            sent_tokens = sent_tokenize(s)

            # get the total number of words and total sentences.
            def total_word_sent():

                words = 0
                sents = 0
                for sent in sent_tokens:
                    words += len([word for word in word_tokenize(sent) if word not in special])
                    sents += 1

                return words, sents

            total_num_of_words, total_num_of_sent = total_word_sent()

            # calculate the vg number of words per sentence for each file and add them to the list.
            average_num_of_words_per_sentences.append(round(total_num_of_words / total_num_of_sent))

        return average_num_of_words_per_sentences

    
    def complex_word_count(self):

        total_complex_words = []

        special = '@_!#$%^&*()<>?/\|}{~:;.[]'

        for file in self.files:
            with open(file, 'r', encoding='utf-8') as f:
                s = f.read()
            word_tokens = word_tokenize(s)

            # remove special characters from the words
            clean_words = [word for word in word_tokens if word not in special]
     
            # complex words are those with syllables > 2
            complex_words = []

            for word in clean_words:
                syllable_count = textstatistics().syllable_count(word)
                if syllable_count > 2:
                    complex_words.append(word)
 
            total_complex_words.append(len(complex_words))
        
  
        return total_complex_words
 



    def word_count(self):

        words_count = []
        
        stop_words = set(stopwords.words('english'))
        special = '@_!#$%^&*()<>?/\|}{~:;.[]'

        for file in self.files:
            with open(file, 'r', encoding='utf-8') as f:
                s = f.read()
            word_tokens = word_tokenize(s)

            # removes the special characters and stop words from the text. 
            clean_stop = [word for word in word_tokens if word not in stop_words]
            clean_words = [word for word in clean_stop if word not in special]

            words_count.append(len(clean_words))

        return words_count


    def syllable_count_per_word(self):

        syllable_count = []

        special = '@_!#$%^&*()<>?/\|}{~:;.[]'

        for file in self.files:
            with open(file, 'r', encoding='utf-8') as f:
                s = f.read()
            word_tokens = word_tokenize(s)

            # remove special characters from the words
            clean_words = [word for word in word_tokens if word not in special]

            syl_count = 0

            for word in clean_words:
                # remove the words ending with 'es' and 'ed' and get the total number of syllables 
                # in each words
                if not word.endswith('es') and not word.endswith('ed'):
                    syllable = textstatistics().syllable_count(word)
                    syl_count += syllable
                
            syllable_count.append(syl_count)

        # returns the number of syllables in each article.
        return syllable_count

    
    def personal_pronoun(self):
        '''the personal pronouns are extracted using regex from re library'''

        personal_pronouns = []

        for file in self.files:
            with open(file, 'r', encoding='utf-8') as f:
                s = f.read()
            
            # finds all the personal pronouns in the text.
            pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b',re.I)
            pronouns = pronounRegex.findall(s)

            personal_pronouns.append(len(pronouns))
            
        # returns the number of personal pronouns found in the text.
        return personal_pronouns


    def average_word_length(self):
        
        # get the total numer of characters in each word and divide it by the total number of words
        # after the special characters have been removed.
        average_word_length = [] 
        special = '@_!#$%^&*()<>?/\|}{~:;.[]'

        for file in self.files:
            with open(file, 'r', encoding='utf-8') as f:
                s = f.read()
            word_tokens = word_tokenize(s)

            clean_words = [word for word in word_tokens if word not in special]

            total_words = len(clean_words)

            # for each word in the clean worrds get the number of characters in the word.
            char_count = 0
            for word in clean_words:
                char_count += len(word)

            average_word_length.append(round(char_count/total_words, 2))

        return average_word_length


def output():

    txt_analysis = TextAnalysis(files, dictionary)
    positive_scores, negative_scores, polarity_scores, subjectivity_scores = txt_analysis.sentiment()
    average_sentence_length, percentage_of_complex_words, fog_index = txt_analysis.readability()
    average_num_of_words_per_sentences = txt_analysis.average_words_per_sentence()
    complex_word = txt_analysis.complex_word_count()
    word_count = txt_analysis.word_count()
    syllable_count = txt_analysis.syllable_count_per_word()
    personal_pronouns = txt_analysis.personal_pronoun()
    average_word_length = txt_analysis.average_word_length()
    ids, links = link.links()

    # dataframe created with all the necessary variables.
    output = pd.DataFrame({'URL_ID': ids, 'URL': links, 'POSITIVE SCORE': positive_scores, 'NEGATIVE SCORE': negative_scores, \
                       'POLARITY SCORE': polarity_scores, 'SUBJECTIVITY SCORE': subjectivity_scores, 'AVG SENTENCE LENGTH': average_sentence_length,\
                       'PERCENTAGE OF COMPLEX WORDS': percentage_of_complex_words, 'FOG INDEX': fog_index, 'AVG NUMBER OF WORDS PER SENTENCE': average_num_of_words_per_sentences, \
                       'COMPLEX WORD COUNT': complex_word, 'WORD COUNT': word_count, 'SYLLABLE PER WORD': syllable_count, \
                       'PERSONAL PRONOUNS': personal_pronouns, 'AVG WORD LENGTH': average_word_length})
    
    # dataframe saved as an excel file.
    output.to_excel('My Output.xlsx')

    return "output file saved as 'My Output.xlsx"

print(output())