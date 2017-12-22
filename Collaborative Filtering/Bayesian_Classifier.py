# @Author: Agasthya Vidyanath Rao Peggerla.
import numpy as np
from pyspark import SparkContext
import numpy as np
import collections
import sys

sc = SparkContext(appName="Bayesian_Filetring")
### Loading Training Dataset
training_data = sc.textFile(sys.argv[1]).map(lambda l:l.split('\t'))

movie_names = sc.textFile(sys.argv[3]).map(lambda l:l.split('|')).map(lambda x: x[1]).collect()

### Grouping the training data based on user
group_user=training_data.groupBy(lambda x:x[0]).sortBy(lambda x:x[0]).collect()

### Separating the fields from training data
### Store user_ids
training_users = training_data.map(lambda x: int(x[0])).collect()

### Store movie_ids
training_movies = training_data.map(lambda x: int(x[1])).collect()

### Store Ratings
ratings = training_data.map(lambda x: int(x[2])).collect()

### Loading the testing datasets.
testing_data = sc.textFile(sys.argv[2]).map(lambda l:l.split('\t'))

### Grouping the training data based on user

test_group_user=testing_data.groupBy(lambda x:x[0]).sortBy(lambda x:x[0]).collect()
### Separating the fields from training data

### Store user_ids
testing_users = testing_data.map(lambda x: int(x[0])).collect()

### Store movie_ids
testing_movies = testing_data.map(lambda x: int(x[1])).collect()

### Store ratings
test_ratings = testing_data.map(lambda x: int(x[2])).collect()

### Finding the maximum user and movie indices in training data
training_users_max= training_data.map(lambda x: int(x[0])).distinct().max()
training_movies_max = training_data.map(lambda x: int(x[1])).distinct().max()

### Finding the maximum user and movie indices in testing data
testing_users_max= testing_data.map(lambda x: int(x[0])).distinct().max()
testing_movies_max = testing_data.map(lambda x: int(x[1])).distinct().max()

###Matrix to store user movie and rating information
train_rating_mat=np.zeros((training_users_max,training_movies_max))
test_rating_mat = np.zeros((testing_users_max,testing_movies_max))

total_ones_train = 0;
total_minus_ones_train = 0;

total_ones_test = 0;
total_minus_ones_test = 0;


##creating matrix of users and movies from training data by populating user provided rating
for i in range(len(group_user)):
    for j in group_user[i][1]:
        if(int(j[2]) >= 3):
            train_rating_mat[int(j[0])-1,int(j[1])-1] = 1 
            total_ones_train+=1
        else:
            train_rating_mat[int(j[0])-1,int(j[1])-1] = -1
            total_minus_ones_train+=1


test_rdd = []

##creating matrix of users and movies from testing data by populating user provided rating
for i in range(len(test_group_user)):
    for j in test_group_user[i][1]:
        if(int(j[2]) >= 3):
            test_rating_mat[int(j[0])-1,int(j[1])-1] = 1
            total_ones_test+=1
            test_rdd.append(1)
        else:
            test_rating_mat[int(j[0])-1,int(j[1])-1] = -1
            total_minus_ones_test+=1
            test_rdd.append(-1)

tot_rats = total_ones_train+total_minus_ones_train

##Calculating the probability of ones in the ratings matrix
total_ones_prob = float(total_ones_train)/tot_rats

## Calculating the probability of -1 in the rating matrix
total_minus_ones_prob = float(total_minus_ones_train)/tot_rats

### Naive Bayes Classifier Definition
def naive_bayes_classifier(line):

    ### Getting the index of the row and column based on the values of user_id and movie_id
    movie_list = train_rating_mat[:, int(line[1])-1]
    user_list = train_rating_mat[int(line[0])-1]

    ### Store the values of the row and column represented by the user_id and movie_id
    m_list = np.array(movie_list)
    u_list = np.array(user_list)

    ## Counter to get total number of 1's and -1's
    movie_counter = collections.Counter(m_list)
    user_counter = collections.Counter(u_list)

    ### Calculating number of -1's
    num_minus_ones_m_list = movie_counter[-1]
    num_minus_ones_u_list = user_counter[-1]

    ### Calculating number of 1's
    num_ones_m_list = movie_counter[1]
    num_ones_u_list = user_counter[1]

    ### Calculating the probability of 1
    prob_ones = float(((num_ones_m_list + num_ones_u_list) + 1))/ (num_ones_u_list + num_ones_m_list + num_minus_ones_m_list + num_minus_ones_u_list + 2)
    prob_ones_tot = float(prob_ones * total_ones_prob)

    ### Calculating the robability of -1
    prob_minus_ones = float(((num_minus_ones_m_list + num_minus_ones_u_list) + 1))/ (num_ones_u_list + num_ones_m_list + num_minus_ones_u_list + num_minus_ones_m_list + 2)
    prob_minus_ones_tot = float(prob_minus_ones * total_minus_ones_prob)

    ### Returing the class label
    if(prob_ones > prob_minus_ones):
        return 1
    else:
        return -1

### Call to naive bayes classifier
predictedRdd = testing_data.map(naive_bayes_classifier).collect()

correct_pred = 0
total_pred = 0

### Calculating the accuracy 
for i in range(0,len(predictedRdd)):
    if(int(predictedRdd[i]) == int(test_rdd[i])):
        correct_pred+=1
total_pred = len(predictedRdd)
accuracy = float(correct_pred)/total_pred

print("__________________________________________________________________")
print("__________________________________________________________________")

### Printing the recommended movies for a user
for i in range(0,25):
    if(int(predictedRdd[i]) ==  1):
        user_id = int(training_users[i])
        movie_id = int(training_movies[i])
        movi_name = movie_names[movie_id]
        print("Recommended movie for user ",user_id," : ",movi_name )


print("Accuracy: ")
print(accuracy*100)


print("__________________________________________________________________")
print("__________________________________________________________________")

