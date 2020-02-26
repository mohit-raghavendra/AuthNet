
def create_multiclass_labels():
	y_train = []
	a = 0
	b = 0
	c = 0
	d = 0
	#for X_train and y_train
	for i in range(no of samples):
		if i>= 0 and i<=9:
			#add it to class 1
			y_train[i] = 1
			
		else 
			if i>= 10 and i <=99:
			#add it to class 
			y_train[i] = 2
			else 
				if i%100 < 10:
					#add it to class 3
					y_train[i] = 3
					else: 
						#add it to class 4  
						y_train[i] = 4

	return y_train



def create_binary_labels():
	
	y_train = []
	#for X_train and y_train
	for i in range(no of samples):
		if i>= 0 and i<=9:
			#add it to class 1
			y_train[i] = 1
		else:
			#add it to class 0
 			y_train[i] = 0

	return y_train
