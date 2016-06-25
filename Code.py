'''
Music Artist Recommender System	

This code is to develop a recommender system that will recommend new musical artists to a user based on their listening history. Suggesting different songs or musical artists to a user is important to many music streaming services, such as Pandora and Spotify. In addition, this type of recommender system could also be used as a means of suggesting TV shows or movies to a user (e.g., Netflix). 
To create this system Spark (using pyspark) and a collaborative filtering technique (Alternative Least Squares) were deployed.
'''
#Code started in a line must end in the same line. The code here is just for representation. Use 
#Recommender.py for execution and it should be run in a spark environment
#Loading required packages

from pyspark.mllib.recommendation import *
import random
import urllib
from operator import *

#Loading data

'''
The data considered was publicly available song data at http://www-etud.iro.umontreal.ca/~bergstrj/audioscrobbler_data.html. However, the original data files were modified so that the code will run in a reasonable time on a single machine. Those reduced files were uploaded into dropbox and used here. 

'''

artistData = sc.parallelize(urllib.urlopen('https://dl.dropboxusercontent.com/u/16867208/Surya_Spark_Files/artist_data_small.txt').read().split('\n')).filter(lambda a: True if a else False).map(lambda a : a.split()).map(lambda b : (int(b[0]), " ".join(b[1:])))
artistAlias = sc.parallelize(urllib.urlopen('https://dl.dropboxusercontent.com/u/16867208/Surya_Spark_Files/artist_alias_small.txt').read().split('\n')).filter(lambda a: True if a else False).map(lambda a : a.split('\t')).map(lambda b : {int(b[0]): int(b[1])})
userArtistData = sc.parallelize(urllib.urlopen('https://dl.dropboxusercontent.com/u/16867208/Surya_Spark_Files/user_artist_data_small.txt').read().split('\n')).filter(lambda a: True if a else False).map(lambda a : a.split()).map(lambda b : (int(b[0]), int(b[1]), int(b[2])))

def mergeDict(a, b) :
    if a :
        a.update(b)
    return a

artistAliasDict = artistAlias.reduce(mergeDict)
for userArtist in userArtistData.collect() :
        if userArtist[1] in artistAliasDict :
                userArtist = (userArtist[0], artistAliasDict[userArtist[1]], userArtist[2])
                break

#Data Exploration

#This is just to see how big is the data and how the statistics are

userPlayData = userArtistData.map(lambda a : (a[0], (a[2], 1)))
def calcPlayCount(a, b) :
        if a :
                data = list(a)
                data[1] = data[1] + 1
                data[0] += b[0]
                a = tuple(data)
        return a

userPlayData = userPlayData.reduceByKey(calcPlayCount).map(lambda a : (a[0], a[1][0], a[1][0]/a[1][1]))
userTop3 = userPlayData.takeOrdered(3, lambda k : -k[1])
for i in range(3) :
        print "User " + str(userTop3[i][0]) + " has a total play count of " + str(userTop3[i][1]) + " and a mean play count of " + str(userTop3[i][2]) + "."

'''
Output:
User 1059637 has a total play count of 674412 and a mean play count of 1878.
User 2064012 has a total play count of 548427 and a mean play count of 9455.
User 2069337 has a total play count of 393515 and a mean play count of 1519.
'''

#Train, Validation and Test Split

trainData, validationData,testData = userArtistData.randomSplit([0.4, 0.4, 0.2], 13)
trainData.cache()
validationData.cache()
testData.cache()
print trainData.take(3)
print validationData.take(3)
print testData.take(3)
print len(trainData.collect())
print len(validationData.collect())
print len(testData.collect())

'''
Output:
[(1059637, 1000049, 1), (1059637, 1000056, 1), (1059637, 1000113, 5)]
[(1059637, 1000010, 238), (1059637, 1000062, 11), (1059637, 1000112, 423)]
[(1059637, 1000094, 1), (1059637, 1000130, 19129), (1059637, 1000139, 4)]
19817
19633
10031
'''
#Function to evaluate the models

def modelEval(model, data) :
    score = 0
    numUsers = 0
    #Find all unique artists in dataset userArtistData : user, artist, rating
    userArtistData_artists = userArtistData.map(lambda a : a[1]).distinct()
    #Format data into user : [artists] structure
    data_formatted = data.map(lambda a : (a[0], a[1])).groupByKey()
    users = data.map(lambda u : u[0]).distinct().collect()
    for user in users :
        #Find the artist list difference
        trainingArtists = trainData.filter(lambda u : u[0] == user).map(lambda a : a[1])
        predictionArtists = userArtistData_artists.subtract(trainingArtists)
        predictions = model.predictAll(sc.parallelize([user]).cartesian(predictionArtists))
        dataArtists = data.filter(lambda u : u[0] == user).map(lambda a : a[1])
        X = len(dataArtists.collect())
        predictionsSorted = predictions.takeOrdered(X, lambda a : -a[2])
        predictionsSorted = sc.parallelize(predictionsSorted).map(lambda a : a[1])
        score += len(dataArtists.intersection(predictionsSorted).collect()) / float(X)
    return score/len(users)

#Model Development and evaluation

ratings = trainData
model1 = ALS.trainImplicit(ratings, 2, seed = 345)
model2 = ALS.trainImplicit(ratings, 10, seed = 345)
model3 = ALS.trainImplicit(ratings, 20, seed = 345)

modelEval1 = modelEval(model1, validationData)
modelEval2 = modelEval(model2, validationData)
modelEval3 = modelEval(model3, validationData)

print "The model score for rank 2 is " + str(modelEval1)
print "The model score for rank 10 is " + str(modelEval2)
print "The model score for rank 20 is " + str(modelEval3)

'''
Output:
The model score for rank 2 is 0.0904308588871
The model score for rank 10 is 0.0952938950541
The model score for rank 20 is 0.0902480866808

'''
#Finding the best model

bestModel = ALS.trainImplicit(trainData, rank=10, seed=345)
modelEval(bestModel, testData)

'''
Output:
0.050731430151914746

'''

#***This is how the program will work**##

top5 = bestModel.recommendProducts(1059637, 5)
for i in range(5) :
    print "Artist %d: %s" % (i, artistData.filter(lambda id : id[0] == top5[i].product).collect()[0][1])

'''
Output:
Artist 0: Brand New
Artist 1: Taking Back Sunday
Artist 2: Evanescence
Artist 3: Elliott Smith
Artist 4: blink-182
'''
