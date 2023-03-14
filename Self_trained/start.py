# Imports
import numpy as np
import pandas as pd
import string
import torch
import torch.nn as nn
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('movie_reviews')

class TheDataset(Dataset):
    def __init__(self, convert_output, test_train):
        self.test_train = test_train

        # Creating a list of movie reviews
        docs = [' '.join(movie_reviews.words(file_id)) for file_id in movie_reviews.fileids()]

        # Creating a list of the associated ground truths
        
        self.cats = [1 if category=='pos' else 0 for file_id in movie_reviews.fileids() for category in movie_reviews.categories(file_id)]
        if convert_output == True:
            self.cats = self.convert(self.cats)
        cleaned_docs = self.preprocess(docs)

        # Uses the CountVectorizer to count each token that occurs at least twice in each document
        count_vectorizer = CountVectorizer(min_df=2)
        X = count_vectorizer.fit_transform(cleaned_docs)

        # Turn the matrix that CountVectorizer creates into a more interpretable dataframe
        self.doc_term = pd.DataFrame(X.todense(), columns=count_vectorizer.get_feature_names_out())
        
        X_train, X_test, y_train, y_test = train_test_split(self.doc_term, self.cats)
        
        self.X_train, self.X_test, self.y_train, self.y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    def __getitem__(self, idx):
        if self.test_train == "train":
            text = self.X_train[idx]
            label = self.y_train[idx]
        else:
            text = self.X_test[idx]
            label = self.y_test[idx]
        return text, label

    def __len__(self):
        if self.test_train == "train":
            return len(self.X_train)
        else:
            return len(self.X_test)
    
    def get_n_input(self):
        n_input = self.X_train.shape[1]
        return n_input

    def preprocess(self, doc_set):
        """
        Input  : docuemnt list
        Purpose: preprocess text (tokenize, removing stopwords, and stemming)
        Output : preprocessed text
        """
        # initialize regex tokenizer
        tokenizer = RegexpTokenizer("(?:(?<=\s)|(?<=^)|(?<=[>\"]))[a-z-']+(?:(?=\s)|(?=\:\s)|(?=$)|(?=[.!,;\"]))")
        # create English stop words list
        en_stop = set(stopwords.words('english'))
        # Create p_stemmer of class PorterStemmer
        p_stemmer = PorterStemmer()
        # list for tokenized documents in loop
        texts = []
        # loop through document list
        for i in doc_set:
            # remove punctuation
            i = i.translate(str.maketrans('', '', string.punctuation))
            i = i.replace('-', '')
            i = i.replace('\'', '')
            # clean and tokenize document string
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]
            # stem tokens
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            # add tokens to list
            texts.append(' '.join(stemmed_tokens))
        return texts

    def convert(self, liste):
        '''
            Takes in list of output classes, converts output classes based on Problem Explenation

            Try:
                1 -> 0.785
                0 -> 0.215
            Input: List of groundtruth outputs
            Output: converted List
        '''
        new_list = [0.785 if i == torch.tensor([1]) else 0.215 for i in liste]
        return new_list

def train(model, dataset, epochs, gpu, learning_rate): # do cuda

    print()
    print("  STARTING TRAINING PROCESS  ")

    # trainloop
    if gpu:
        loss_function = nn.BCELoss().to('cuda')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        loss_function = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    for epochs in tqdm(range(epochs)):
        for data in dataset:
            x, y = data
            if gpu:
                x.to('cuda')
                y.to('cuda')

            # forward
            pred_y = model(x.type(torch.FloatTensor))

            # loss
            loss = loss_function(pred_y.view(-1).type(torch.FloatTensor), y.type(torch.FloatTensor))
            
            # Append loss
            losses.append(loss.item())

            # Backprop
            model.zero_grad()
            loss.backward()
            optimizer.step()
    return losses

def test(model, dataset):

    print()
    print("  STARTING TEST PROCESS  ")

    model.eval()
    predictions = []
    groundtruth = []
    with torch.no_grad():
        for data in tqdm(dataset):
            predictions.append(model(data[0].type(torch.FloatTensor)))
            groundtruth.append(data[1].type(torch.FloatTensor))
    
    a = []
    b = []
    for idx in range(len(predictions)):
        a.append(predictions[idx].item())
        b.append(groundtruth[idx].item()) 


    return a, b

# Hyperparameters
gpu = False
convert_output = False
save_model_flag = "normal_model_1000_epochs_01_lr_136_batch"    # if empty string: doesn't save model, if not empty string: saves model in output directory with given name
output_path = "./output/"
training_epochs = 1000
learning_rate = 0.01
batch_size = 136
n_hidden = 10000
n_out = 1

if __name__ == "__main__":
    
    # initialize dataset
    train_set = TheDataset(convert_output, "train")
    test_set = TheDataset(convert_output, "test")

    # initialize dataloader
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, 1, shuffle=True)

    # initialize model
    if gpu:
        model = nn.Sequential(nn.Linear(train_set.get_n_input(), n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, n_out),
                    nn.Sigmoid()).to('cuda')
    else:
        model = nn.Sequential(nn.Linear(train_set.get_n_input(), n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, n_out),
                    nn.Sigmoid())

    losses = train(model, train_loader, training_epochs, gpu, learning_rate)
    predictions_on_test = test(model, test_loader)

    if not(save_model_flag == False):
        torch.save(model.state_dict(), output_path + save_model_flag)
