#create a new python file and import the following packages
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

#define a function and extract features
def extract_features(word_list):
    return dict([(word,True) for word in word_list])

#we need training data set for this, so we will use movie reviews in NLTK
if __name__=='__main__':
    
    # load positive and negative reviews
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')
    
    #separate into positive and negative reviews
    features_positive = [(extract_features(movie_reviews.words(fileids=[f])),'Positive') for f in positive_fileids]
    features_negative = [(extract_features(movie_reviews.words(fileids=[f])),'Negative') for f in negative_fileids]
    
    #split the dataset into train and test (80/20)
    threshold_factor = 0.8
    threshold_positive = int(threshold_factor * len(features_positive))
    threshold_negative = int(threshold_factor * len(features_negative))
    
    #extract the features
    features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
    features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]  
    
    #we will use a Naive Byes Classifier. define the object and train it
    classifier = NaiveBayesClassifier.train(features_train)
    features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]  
    
    #the classifier object contains the most informative words that it obtained analysi. these words basically have a strong say in what's classified as a positive or negative review.
    print ("\nTop 10 most informative words:")
    for item in classifier.most_informative_features()[:10]:
        print (item[0])
        
    #create a couple random input sentences
    #sample input reviews
    input_reviews = [" It is amazing movie",
                     "This is a dull movie. I would never recommend it to anyone.",
                     "The cinematography is pretty great in the movie",
                     "The direction was terrible and the story was all over the place"]
    
    #run the classifier on the input sentences and obtain the predictions 
    print("\nPredictions:")
    for review in input_reviews:
        print("\nReview:", review)
        probdist = classifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probdist.max()
        
        #print the output
        print("Predicted sentiment:", pred_sentiment)
        print("Probability:", round(probdist.prob(pred_sentiment)), 2)

    
    