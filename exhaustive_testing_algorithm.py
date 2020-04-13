
accuracy_list = []
training_people = [P1, P2 ... Pn] #Whoever ever people we are training and testing on, write their personID
word_list = [W1, W2, ... W10]
testing_people = [P1, P2, P3]

#iterate through each person
for personID in training_people:

	person_accuracy = []

	#iterate through each word
	for wordID in word_list:

		#Set the appropriate word as positive and rest as negative

		#create training set and testing set
		
		X_train = concatenator(personID, wordID) 
		X_train_oversampled = oversample(X_train)

		y_train = labeller(personID, wordID)

		X_train,X_test,y_train,Y_test=train_test_split(X_train_oversampled, y_train, test_size=0.1,random_state=42)

		#Train the model for 60 iterations 

		model=trainer(X_train,y_train)

		#Test it for the test set(including 3 other people)
		testing_set=concatenator_t(testing_people)
		testing_set=testing_set.reshape(300,20,2622)

		X_test_final=np.concatenate((X_test,testing_set))
		y_test_final=np.concatenate((Y_test,np.zeros((300,1))))
		from sklearn.metrics import accuracy_score	
		accuracy = accuracy_score(y_test_final,y_pred)

		#record accuracy for that person and word

		person_accuracy.append(accuracy)


	accuracy_list.append(person_accuracy)