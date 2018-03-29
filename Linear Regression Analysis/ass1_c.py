from pandas import DataFrame   # To get R-like functionality - to strore data in the form of dataframe, for convenient handling, as in R
from pandas import read_csv	 	# to read data from csv file into a DataFrame
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

path='kc_house_data.csv'

def data_reader(file_path):
	data=read_csv(file_path)
	n_rows=data.shape[0]
	n_cols=data.shape[1]

	train_set_count=int(n_rows*0.8)

	train_param=data.iloc[:train_set_count,:n_cols-1]      # stores the training set's attributes alone
	train_target=data.iloc[:train_set_count,n_cols-1]  # stores the training set's target values

	test_param=data.iloc[train_set_count:,:n_cols-1]      # stores the training set's attributes alone
	test_target=data.iloc[train_set_count:,n_cols-1]  # stores the training set's target values

	return train_param,train_target,test_param,test_target



# retuns the partial differential for the MSE cost function- for the provided theta_vector and the x_vector
# expects a everything as numpy arrays and matrices
def mse_partial_differential(theta_vector,x_vectors,y_values):              
	predictions=np.dot(x_vectors,theta_vector)
	Y=np.subtract(predictions,y_values)
	Y=np.dot(Y.transpose(),x_vectors)
	return np.divide(Y,x_vectors.shape[0])    

# since the default value of regularization_param is 0, the function uses unregularized version by default
def grad_desc(learn_rate,cost_func_type,x_vectors,y_values,regularization_param=0):	
	n_features=x_vectors.shape[1]
	theta_vector= np.random.rand(n_features,1)   # 5 rows; 1 col
	partial_derivative=0

	iteration_count=0
	difference_ratio=1

	old_cost_value=1
	cost_value=0

	
	#(:,np.newaxis)
	while difference_ratio>0.0001:
		new_theta_vector=np.zeros(theta_vector.shape)
		if cost_func_type=='MSE':
			partial_derivative=mse_partial_differential(theta_vector,x_vectors,y_values)	
			partial_derivative=partial_derivative.transpose()

		for j in range(theta_vector.shape[0]):
			# not using regularization for theta_0	
			if(j!=0):	
				new_theta_vector[j,0]=theta_vector[j,0]*(1.0-learn_rate*(float(regularization_param)/float(x_vectors.shape[0])))-learn_rate*partial_derivative[j,0]
			if(j==0):
				new_theta_vector[j,0]=theta_vector[j,0]-learn_rate*partial_derivative[j,0]
			#END FOR

			
		theta_vector=new_theta_vector
		
		predictions=get_predictions(theta_vector,x_vectors)
		cost_value=get_mse(predictions,y_values,theta_vector=theta_vector,regularization_param=regularization_param)
		if iteration_count==0:
			old_cost_value=cost_value
			difference_ratio=1
		else:
			difference_ratio=(old_cost_value-cost_value)/old_cost_value
			old_cost_value=cost_value
		iteration_count+=1

	print "\t#iterations: ",iteration_count
	return theta_vector


# Z-NORMALIZATION FUNCTION: takes numpy matrices. assumes that the matrix is passed in the row-major form...
#...-ie, as an array of rows of the matrix
# start and end col give additional parameters to specify which all columns are to be normalized
# M must be a matrix (NOT array), with non-null rows and cols
def Z_norm(M,start_col=1,end_col=None,norm_param=None):
	
	if norm_param is None:
		mean_vector=np.mean(M,axis=0)            
		std_dev_vector=np.std(M,axis=0)
	else:
		mean_vector=norm_param[0]
		std_dev_vector=norm_param[1]

	if end_col is None:
		end_col=M.shape[1]

	for i in range(start_col,end_col):    # not doing the operation for x_0
		M[:,i]=(M[:,i]-mean_vector[i])/std_dev_vector[i]

	if norm_param is  None:
		return M,(mean_vector,std_dev_vector)
	else:
		return M

# if test_param be: nX5, learned params should be 5X1 => output: nX1 (coulumn matrix)
def get_predictions(learned_params,test_param):
	predictions=np.dot(test_param,learned_params)
	return predictions


# takes both parameters as np arrays of COLUMN TYPE  only
def get_mse(test_predictions,test_target,theta_vector=None,regularization_param=0,use_RMSE=False):
	mse=0
	if test_predictions.shape!=test_target.shape:
		print "ERROR: get_mse(): incompatible arrays",test_predictions.shape,test_target.shape
		exit()
	A=np.subtract(test_predictions,test_target)
	A=np.square(A)
	mse=np.sum(A)

	if not use_RMSE:
		if theta_vector is not None:
			x=np.square(theta_vector)
			x=np.sum(x)
			mse+=regularization_param*x
		mse=float(mse)/float(A.shape[0])
		return mse/2
	else:
		mse=float(mse)/float(A.shape[0])
		return sqrt(mse)


def main():

	n_features=0

	train_param,train_target,test_param,test_target=data_reader(path)
	n_features=train_param.shape[1]+1   # x_0 to be inserted yet
	train_param.insert(0,'x_0',[1]*train_param.shape[0])
	test_param.insert(0,'x_0',[1]*test_param.shape[0])

	train_param=train_param.as_matrix()
	test_param=test_param.as_matrix()
	train_target=train_target.as_matrix()
	test_target=test_target.as_matrix()
	train_target=train_target[:,np.newaxis]
	test_target=test_target[:,np.newaxis]

	# norm_param is the tuple (mu,std_dev)- to store the mean and the standard deviation ...
	# ...of the training set, for each feature (other than x_0)
	
	train_target,norm_param_target=Z_norm(train_target,start_col=0,end_col=1)
	test_target=Z_norm(test_target,start_col=0,end_col=1,norm_param=norm_param_target)




	learning_rates=[0.001,0.002,0.005,0.008,0.01,0.03,0.05,0.08,0.1,0.2]#,0.3,0.5]#

	# for LINEAR COMBINATIONS:
	print "With Linear combination of features : \n"

	# NORMALIZING
	new_train_param,norm_param=Z_norm(train_param)             # (x-mu)/sigma
	new_test_param=Z_norm(test_param,norm_param=norm_param)  # normalizing the test parameters according to the parameters(mu,sigma) obtained for the train set


	errors_lin=[]

	for rate in learning_rates:
		print"\tLearning Rate: ",rate
		learned_params=learned_params=grad_desc(rate,'MSE',new_train_param,train_target)
		test_predictions=get_predictions(learned_params,new_test_param)
		error=get_mse(test_predictions,test_target,use_RMSE=True)
		errors_lin.append(error)
		print "\tLearned Parameters: ",learned_params.transpose()
		print "\tRMSE: ",error
	#print errors_lin



	# creating ALL QUADRATIC COMBINATIONS- (x_0,x_0),(x_0,x_1)....(x_2,x_2),(x_2,x_3).. and so on   :
	# NOTE : as x_0=1, this methhid ensures that ALL ORIGINAL FEATURES ARE PRESENT themselves as well
	print "\n\n\nQuadratic Combinations: UPTO 2nd ORDER TERMS TAKEN: the considered features are the features themselves, their squares and products of 2 taken at a time"

	all_feature_combinations=list(combinations_with_replacement(range(n_features),2))
	new_train_param=np.zeros((train_param.shape[0],len(all_feature_combinations)))
	new_test_param=np.zeros((test_param.shape[0],len(all_feature_combinations)))
	for i in range(len(all_feature_combinations)):
		col_1=all_feature_combinations[i][0]
		col_2=all_feature_combinations[i][1]
		new_train_param[:,i]=np.multiply(train_param[:,col_1],train_param[:,col_2])
		new_test_param[:,i]=np.multiply(test_param[:,col_1],test_param[:,col_2])	
	# end for

	new_train_param,norm_param=Z_norm(new_train_param)             # (x-mu)/sigma
	new_test_param=Z_norm(new_test_param,norm_param=norm_param)  # normalizing the test parameters according to the parameters(mu,sigma) obtained for the train set

	errors_quad=[]
	for rate in learning_rates:
		print"\tLearning Rate: ",rate
		learned_params=learned_params=grad_desc(rate,'MSE',new_train_param,train_target)
		test_predictions=get_predictions(learned_params,new_test_param)
		error=get_mse(test_predictions,test_target,use_RMSE=True)
		errors_quad.append(error)
		print "\tLearned Parameters: ",learned_params.transpose()
		print "\tRMSE: ",error
	#print errors_quad




	# creating ALL CUBIC COMBINATIONS- (x_0,x_0,x_0),(x_0,x_0,x_1)....(x_2,x_2,x_2),(x_2,x_2,x_3).. and so on   :
	# NOTE : as x_0=1, this methhid ensures that ALL ORIGINAL FEATURES ARE PRESENT themselves as well
	print "\n\n\nCubic Combinations: UPTO 3rd ORDER TERMS TAKEN: the considered features are the features themselves, their squares,cubes and products of 3 taken at a time"



	all_feature_combinations=list(combinations_with_replacement(range(n_features),3))
	new_train_param=np.zeros((train_param.shape[0],len(all_feature_combinations)))
	new_test_param=np.zeros((test_param.shape[0],len(all_feature_combinations)))
	for i in range(len(all_feature_combinations)):
		col_1=all_feature_combinations[i][0]
		col_2=all_feature_combinations[i][1]
		col_3=all_feature_combinations[i][2]
		new_train_param[:,i]=np.multiply(train_param[:,col_1],train_param[:,col_2])
		new_train_param[:,i]=np.multiply(new_train_param[:,i],train_param[:,col_3])
		new_test_param[:,i]=np.multiply(test_param[:,col_1],test_param[:,col_2])
		new_test_param[:,i]=np.multiply(new_test_param[:,i],test_param[:,col_3])	
	# end for

	new_train_param,norm_param=Z_norm(new_train_param)             # (x-mu)/sigma
	new_test_param=Z_norm(new_test_param,norm_param=norm_param)  # normalizing the test parameters according to the parameters(mu,sigma) obtained for the train set


	errors_cubic=[]
	for rate in learning_rates:
		print"\tLearning Rate: ",rate
		learned_params=learned_params=grad_desc(rate,'MSE',new_train_param,train_target)
		test_predictions=get_predictions(learned_params,new_test_param)
		error=get_mse(test_predictions,test_target,use_RMSE=True)
		errors_cubic.append(error)
		print "\tLearned Parameters: ",learned_params.transpose()
		print "\tRMSE: ",error
	#print errors_cubic


	l1,l2,l3=plt.plot(learning_rates,errors_lin,learning_rates,errors_quad,learning_rates,errors_cubic)
	plt.setp(l1,linewidth=3,color='r')
	plt.setp(l2,linewidth=2,color='g')
	plt.setp(l3,linewidth=2,color='b')
	plt.legend([l1,l2,l3],['Linear','Quadratic','Cubic'])
	plt.xlabel('learning rate')
	plt.ylabel('RMSE')
	#plt.xticks(np.arange(min(learning_rates), max(learning_rates)+1, 0.05))
	plt.savefig("ass1_c.png")
	

main()