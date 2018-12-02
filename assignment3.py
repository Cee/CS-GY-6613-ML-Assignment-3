import numpy as np

class KNN:
	def __init__(self, k):
		#KNN state here
		#Feel free to add methods
		self.k = k

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		#training logic here
		#input is an array of features and labels

		# As mentioned in the instruction,
		# training here is just memorizing the data
		self.features = X
		self.labels = y

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		prediction = np.array([])
		for x in X:
			# Calculate distance with each feature
			distances = np.array([self.distance(x, f) for f in self.features])
			# Zip with (distance, label)
			neighbours = zip(distances, self.labels)
			# Get sorted, pick top k
			top_k_neighbours = sorted(neighbours, key=lambda x: x[0])[:self.k]
			# Prediction
			predict = self.majority([x[1] for x in top_k_neighbours])
			# Append predict to ans
			prediction = np.append(prediction, [predict])	
		return prediction

	def majority(self, neighbours):
		count = 0
		candidate = None
		for n in neighbours:
			if count == 0:
				candidate = n
			count += (1 if n == candidate else -1)
		return candidate

class ID3:
	class Node(object):
		def __init__(self):
			self.attribute = None # key
			self.values = [] # mapping key -> values
			self.children = [] # samples
			self.label = None # if pure, mark 0 or 1

	def __init__(self, nbins, data_range):
		#Decision tree state here
		#Feel free to add methods
		self.bin_size = nbins
		self.range = data_range

	def preprocess(self, data):
		#Our dataset only has continuous data
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data

	def train(self, X, y):
		#training logic here
		#input is array of features and labels
		categorical_data = self.preprocess(X)
		# Mark attributes with 0 - (col - 1)
		columns = np.size(categorical_data, 1)
		attributes = np.arange(columns)
		self.root = self.decision_tree_learning(categorical_data, attributes, None, y)

	def decision_tree_learning(self, examples, attributes, parent_examples, y):
		node = self.Node()
		# 1) Examples is empty
		if len(examples) == 0:
			node.label = self.plurality_value(parent_examples)
		# 2) Same classification
		elif len(np.unique(y)) == 1:
			node.label = y[-1]
		# 3) Attributes is empty
		elif len(attributes) == 0:
			node.label = self.plurality_value(y)
		# 4) Otherwise, pick the most important value
		else:
			A = np.argmax([self.gain(examples[:,attr], y) for attr in attributes])
			node.attribute = A
			# Generate nodes by value
			for value in np.unique(examples[:,A]):
				child_examples = examples[np.argwhere(examples[:,A] == value).ravel()]
				remain_attributes = np.delete(attributes, A)
				child_y = y[np.argwhere(examples[:,A] == value).ravel()]
				subtree = self.decision_tree_learning(child_examples, remain_attributes, y, child_y)
				node.values.append(value)
				node.children.append(subtree)
		return node

	def entropy(self, p):
		return -sum(x * np.log2(x) for x in p.values())

	def gain(self, x, y):
		p = self.prob(x)
		info = 0
		for key in p.keys():
			i = np.argwhere(x == key).ravel()
			info = info + p[key] * self.entropy(self.prob(y[i]))
		return self.entropy(self.prob(y)) - info

	def prob(self, x):
		values = np.unique(x)
		p = {}
		for v in values:
			p[v] = (x == v).sum() / x.size
		return p

	def plurality_value(self, y):
		return self.majority(y)

	def majority(self, neighbours):
		count = 0
		candidate = None
		for n in neighbours:
			if count == 0:
				candidate = n
			count += (1 if n == candidate else -1)
		# print("candidate:", candidate)
		return candidate

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)
		prediction = np.array([])
		for x in categorical_data:
			# Append each prediction to answer
			prediction = np.append(prediction, [self.traverse(self.root, x)])
		# print(prediction)
		return prediction

	def traverse(self, node, x):
		if node.label != None:
			return node.label
		value = x[node.attribute]
		for i in range(len(node.values)):
			if node.values[i] == value:
				return self.traverse(node.children[i], x)
		return -1

class Perceptron:
	def __init__(self, w, b, lr):
		#Perceptron state here, input initial weight matrix
		#Feel free to add methods
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		#training logic here
		#input is array of features and labels
		# print(self.w)
		# for t in range(steps):
		while steps > 0:
			error = False
			for i, x in enumerate(X):
				y_i = 1 if np.dot(self.w, X[i]) + self.b > 0 else -1
				d_i = 1 if y[i] == 1 else -1
				if not (y_i == d_i):
					self.w = self.w + self.lr * d_i * X[i]
					self.b = self.b + self.lr * d_i
					error = True
					steps -= 1
			# makes no error on training set, break
			if not error:
				break
		# print(self.w)

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		prediction = np.array([])
		for x in X:
			prediction = np.append(prediction, [1 if (np.dot(x, self.w) + self.b) > 0 else 0])	
		return prediction

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi) 
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)

class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, input):
		#Write forward pass here
		self.x = input
		return np.dot(self.x, self.w) + self.b

	def backward(self, gradients):
		#Write backward pass here
		w_prime = np.dot(self.x.transpose(), gradients)
		x_prime = np.dot(gradients, self.w.transpose())
		self.w = self.w - self.lr * w_prime
		self.b = self.b - self.lr * gradients
		return x_prime

class Sigmoid:

	def __init__(self):
		None

	def forward(self, input):
		#Write forward pass here
		self.sigmoid = np.array([1 / (1 + np.exp(-x)) for x in input])
		return self.sigmoid

	def backward(self, gradients):
		#Write backward pass here
		return gradients * np.array([(1 - x) * x for x in self.sigmoid])