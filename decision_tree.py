import numpy as np
#decision_tree from scratch using the ID3 algorithm.
#Assuming the input data is in the form of a dictionary 
#                       Col1    Col2    Col3      
#[	 
#	 [C1R1, C2R1, C3R1, ........], 
# 	 [C1R2, C2R2, C3R2, ........],
#	 .
#           .
#	'[C1RN, C2RN, C3RN, ........]
#]
#The coln represents the columns and the dictionary contains the rows, with 
#keys as the index.
#The last column represents the labels, as we are dealing with a supervised problem.
#The example with which we'll test will take integer values, we can further scale it to dealing 
#with categorical values.
def divide_data(data, column, value):
	"""
	returns a dataset which is split.
	"""
	split_function = None 
	if isinstance(value, int) or isinstance(value, float):
		split_function = lambda row : row[column] < value
	elif isinstance(value, basestring):
		split_function = lambda row : row[column] == value
	else:
		print "TypeError : Unsupported value type."
	subset1 = [row for row in data if split_function(row)]
	subset2 = [row for row in data if not split_function(row)]			
	return (subset1, subset2)

def entropy(data):
	"""
	For caluculating the cross-entropy of the data, using the class proportions 
	Also known as Shannon's Entropy.
	:params: label data as `label`
	:returns: Entropy of all the classes as a list
	"""
	unique_labels = {}
	for _, row in enumerate(data):
		if row[len(row) - 1] not in unique_labels: 
			unique_labels[row[len(row) - 1]] = 0
		unique_labels[row[len(row) - 1]] += 1 
	proportions = [float(i)/len(data) for i in unique_labels.values()]	
	entropy = sum(-p*np.log2(p) for p in proportions)
	return entropy

def information_gain(data, column, cut_point):
	"""
	For calculating the goodness of a split. The difference of the entropy of parent and weighted 
	the weighted entropy of children.
	i.e. I(t) = i1(t)p1()
	:params:attribute_index, labels of the node t as `labels` and cut point as `cut_point`
	:returns: The net entropy of partition 
	"""
	subset1, subset2 = divide_data(data, column, cut_point)            
	lensub1, lensub2 = len(subset1), len(subset2)
	weighted_ent = (len(subset1)*entropy(subset1) + len(subset2)*entropy(subset2)) / len(data)
	if len(subset1) > 0 and len(subset2) > 0:
		return (entropy(data) - weighted_ent, subset1, subset2)
	else:
	 	return 0	

class decision_tree():
	#Every node will have a label
	parent = None
	attribute = None 
	cut_point = None 
	left = None
	right = None

	def __init__(self, parent = None, attribute = None, cut_point = None, left = None, right = None):
		self.label_index = label_index

	def build_tree(self, rows, fitness_function = information_gain):
		if len(rows) == 0: return decision_tree()	

		best_gain = 0
		best_criteria = None
		best_sets = None
		for column in range(len(rows[0] - 1)):
			#the values to try splitting on
			unique_vals = {}
			for row in rows:
				unique_vals[row[column]] = 1
			#Now try to split at this column using every unique value it contains 	
			for uv in  unique_vals.keys():
				gain, sub1, sub2 = fitness_function(rows, column, uv)	
				if gain > best_gain:
					best_gain = gain
					best_criteria = (column, uv)
					best_sets = (sub1, sub2)
		#Now at this point, in the first call our decision tree has experienced its first split 
		#Now proceed only if best gain is worthful
		if best_gain > 0:
			left = build_tree(sub1)	
			right = build_tree(sub2)		
			re 

#COMPLETE THE BASIC STRUCTURE TOMORROW and they try hypothesis testing.			
		