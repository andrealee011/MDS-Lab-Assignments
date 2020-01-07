# Assignment from DSCI 512 Lab 1

class MarkovModel:
    
    def __init__(self, n):
        
        self.n = n
        self.freq_dic = {}
        self.corp = None
    
    def fit(self, corpus):
        
        '''calculates and stores the frequencies of all possible next characters

        Arguments:
        corpus -- input text
        '''
        
        self.corp = corpus + corpus[0:self.n]
        next_char_dic = {}

        start = 0
        end = self.n

        for i in self.corp:

            if end > len(self.corp)-1:
                break

            n_gram = self.corp[start:end]
            next_char = self.corp[end]

            if n_gram not in self.freq_dic:
                next_char_dic[next_char] = 1
                self.freq_dic[n_gram] = next_char_dic

            elif n_gram in self.freq_dic:
                next_char_dic = self.freq_dic[n_gram]
                if next_char not in next_char_dic:
                    next_char_dic[next_char] = 1
                    self.freq_dic[self.corp[start:end]] = next_char_dic
                elif next_char in next_char_dic:
                    next_char_dic[next_char] = next_char_dic[next_char] + 1
                    self.freq_dic[self.corp[start:end]] = next_char_dic

            next_char_dic = {} 
            start = start + 1
            end = end + 1
            
        for i in self.freq_dic:
            total = 0
            next_char_dic = self.freq_dic[i]
            for j in next_char_dic:
                total += next_char_dic[j]
            for j in next_char_dic:
                prob = next_char_dic[j]/total
                next_char_dic[j] = prob
            self.freq_dic[i] = next_char_dic

        return self.freq_dic
        
    def generate(self, length):

        '''creates a random text of a specified length

        Arguments:
        length -- length of the text to be generated
        '''
        
        text = self.corp[0:self.n]
        start = 0
        end = self.n

        for i in self.corp[:length]:

            if end > len(self.corp[:length])-1:
                break

            n_gram = text[start:end]
            next_char_dic = self.freq_dic[n_gram]
            next_char = np.random.choice(list(next_char_dic.keys()),1,list(next_char_dic.values()))
            next_char = next_char[0]
            text += next_char

            start += 1
            end += 1

        return text
        
data_url = 'http://www.gutenberg.org/files/20748/20748.txt'
snow_white = urllib.request.urlopen(data_url).read().decode("utf-8")
snow_white = snow_white[1903:18164]

mm = MarkovModel(n=9)
mm.fit(snow_white)
print(mm.generate(1000))
