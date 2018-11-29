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

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)
		return None

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
		
		# map 1 to 1, 0 to -1
		d = np.array([])
		for x in y:
			d = np.append(d, [1 if x == 1 else -1])
		# print(self.w)
		# for t in range(steps):
		while steps > 0:
			for i, x in enumerate(X):
				if (np.dot(X[i], self.w) + self.b[0]) * d[i] <= 0:
					self.w = self.w + self.lr * X[i] * y[i]
					self.b = self.b + self.lr * y[i]
					steps -= 1
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