from pandas import DataFrame   # To get R-like functionality - to strore data in the form of dataframe, for convenient handling, as in R
from pandas import read_csv	 	# to read data from csv file into a DataFrame
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

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
def partial_differential(theta_vector,x_vectors,y_values,func='MSE'):
	if func=='MSE':              
		predictions=np.dot(x_vectors,theta_vector)
		Y=np.subtract(predictions,y_values)
		Y=np.dot(Y.transpose(),x_vectors)
		return np.divide(Y,x_vectors.shape[0])   
	if func=='MAE':
		predictions=np.dot(x_vectors,theta_vector)
		Y=np.subtract(predictions,y_values)
		sign_vector=np.sign(Y)							#storing the signs of the individual terms: h(x)-y
		Z=np.multiply(x_vectors,sign_vector)
		Z=np.divide(Z,x_vectors.shape[0])
		Z=np.sum(Z,axis=0)
		Z=Z[:,np.newaxis]
		return Z.transpose()

	if func=='MCE':
		predictions=np.dot(x_vectors,theta_vector)
		Y=np.subtract(predictions,y_values)
		Y=np.square(Y)
		Y=np.dot(Y.transpose(),x_vectors)
		return np.divide(Y,x_vectors.shape[0]) 

 
'''
Below is a more generalized version of the gradient descent function. It takes, along with other arguments,
an additional argument indicating the type of cost function to be used and simply passes this value to the
get_error() and the partial_differential() functions.

These, functions, in turn, return the values based on the requested cost function

'''

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
	while difference_ratio>0.00000001:
		new_theta_vector=np.zeros(theta_vector.shape)	
		partial_derivative=partial_differential(theta_vector,x_vectors,y_values,func=cost_func_type)	
		partial_derivative=partial_derivative.transpose()

		for j in range(theta_vector.shape[0]):
			new_theta_vector[j,0]=theta_vector[j,0]-learn_rate*partial_derivative[j,0]
		
		#END FOR			
		theta_vector=new_theta_vector
		
		predictions=get_predictions(theta_vector,x_vectors)
		cost_value=get_cost(predictions,y_values,func=cost_func_type,theta_vector=theta_vector,regularization_param=regularization_param)
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


'''

The most general cost function- returns the cost according to the REQUESTED TYPE OF COST FUNCTIoN

'''
# takes both parameters as np arrays of COLUMN TYPE  only

def get_cost(test_predictions,test_target,func='MSE',theta_vector=None,regularization_param=0,use_RMSE=False):

	if test_predictions.shape!=test_target.shape:
		print "ERROR: get_mse(): incompatible arrays",test_predictions.shape,test_target.shape
		exit()

	if func=='MSE':
		mse=0
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


	# for the mean absolute error: 

	if func=='MAE':
		mae=0
		A=np.subtract(test_predictions,test_target)
		A=np.absolute(A)
		mae=np.sum(A)
		mae=float(mae)/float(A.shape[0])
		return mae


	# for the mean cube error: 

	if func=='MCE':
		mce=0
		A=np.subtract(test_predictions,test_target)
		A=np.power(A,3)
		mce=np.sum(A)
		mce=float(mce)/(3*float(A.shape[0]))
		return mce



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
	train_param,norm_param=Z_norm(train_param)             # (x-mu)/sigma
	test_param=Z_norm(test_param,norm_param=norm_param)  # normalizing the test parameters according to the parameters(mu,sigma) obtained for the train set
	train_target,norm_param_target=Z_norm(train_target,start_col=0,end_col=1)
	test_target=Z_norm(test_target,start_col=0,end_col=1,norm_param=norm_param_target)


	learning_rates=[0.001,0.002,0.005,0.008,0.01,0.03,0.05,0.08,0.1]


	#using different cost_functions

	# the Mean Squared Error cost function:
	print"Using the Mean Squared Error cost function:"
	errors_mse=[]
	for rate in learning_rates:
		print "\tlearning rate: ",rate
		learned_params=learned_params=grad_desc(rate,'MSE',train_param,train_target)
		test_predictions=get_predictions(learned_params,test_param)
		error=get_cost(test_predictions,test_target,use_RMSE=True)
		errors_mse.append(error)
		print "\tlearned parameters: ",learned_params.transpose()
		print "\tRMSE: ",error
	print errors_mse


	# the Mean Absolute Error cost function

	errors_mae=[]
	print"\n\nUsing the Mean Absolute Error cost function:"
	for rate in learning_rates:
		print "\tlearning rate: ",rate
		learned_params=learned_params=grad_desc(rate,'MAE',train_param,train_target)
		test_predictions=get_predictions(learned_params,test_param)
		error=get_cost(test_predictions,test_target,use_RMSE=True)                        # taking the root mean squared error
		errors_mae.append(error)
		print "\tlearned parameters: ",learned_params.transpose()
		print "\tRMSE: ",error
	print errors_mae


	# the Mean Cube Error cost function:

	errors_mce=[]
	print"\n\nUsing the Mean Cube Error cost function:"
	for rate in learning_rates:
		print "\tlearning rate: ",rate
		learned_params=learned_params=grad_desc(rate,'MCE',train_param,train_target)
		test_predictions=get_predictions(learned_params,test_param)
		error=get_cost(test_predictions,test_target,use_RMSE=True)                    
		errors_mce.append(error)
		print "\tlearned parameters: ",learned_params.transpose()
		print "\tRMSE: ",error
	print errors_mce



	l1=plt.plot(learning_rates,errors_mse)#,learning_rates,errors_mae)#,learning_rates,errors_mce)
	plt.setp(l1,linewidth=3,color='r')
	plt.legend(l1,['MSE'])
	#plt.setp(l2,linewidth=2,color='g')
	#plt.setp(l3,linewidth=2,color='b')
	#plt.legend([l1,l2,l3],['MSE','MAE','MCE'])
	plt.xlabel('learning rate')
	plt.ylabel('RMSE')
	plt.savefig("ass1_d_mse.png")
	#plt.show()
	plt.clf()

	l2=plt.plot(learning_rates,errors_mae)#,learning_rates,errors_mae)#,learning_rates,errors_mce)
	plt.setp(l1,linewidth=3,color='g')
	plt.legend(l2,['MAE'])
	plt.xlabel('learning rate')
	plt.ylabel('RMSE')
	#plt.show()
	plt.savefig("ass1_d_mae.png")	
	plt.clf()
	
	l3=plt.plot(learning_rates,errors_mce)#,learning_rates,errors_mae)#,learning_rates,errors_mce)
	plt.setp(l1,linewidth=3,color='b')
	plt.legend(l3,['MCE'])
	plt.xlabel('learning rate')
	plt.ylabel('RMSE')
	#plt.show()
	plt.savefig("ass1_d_mce.png")	
	

main()