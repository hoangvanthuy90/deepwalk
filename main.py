import pandas as pd
import numpy as np
import random
import networkx as nx
from matplotlib import pyplot as plt 
np.random.seed(15)


#Load data
adjlist = nx.read_adjlist("karate_club.adjlist", nodetype=int)
karate_label = np.loadtxt("karate_label.txt")
Graph = nx.read_adjlist("karate_club.adjlist", nodetype=int)
node_number = nx.to_pandas_adjacency(Graph).columns


adj = nx.to_numpy_array(adjlist)
#label = karate_label[:,-1]

print(adj.shape)
#print(label.shape)

#actiavtion function
def softmax(x):
    c = np.max(x)
    b = x-c
    exp_x = np.exp(b)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

# fully conneted network
class ann:
    def __init__(self, input_size, hidden_size, output_size):
        # setting the shape of the layer and putting random inital value
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size)

    # Calculating the Values
    def gradient(self, x, y):
        # forward
        W1, W2 = self.params['W1'], self.params['W2']
        h = np.dot(x, W1)
        # H = softmax(U)

        U2 = np.dot(h, W2)
        Y = softmax(U2)

        diff = (Y - y)  # 34*1

        ERR2 = np.outer(h, diff)

        # backpropagation
        # ERR2 = (-np.log(np.abs(Y-y)))*Y*(1-Y)
        ERR = np.outer(x, np.dot(W2, diff))

        return ERR, ERR2, diff, Y

w = 3 #window_size w
d = 4 #embedding size d
r = 10 # walks per vertex
t = 10 # walk length
learning_rate = 0.0001

#Params
n_network = ann(input_size = 34,hidden_size = d,output_size = 34)
#P = np.random.random((34,d)) # Work as W1 (input_size,hidden_size)
#Q = np.random.random((d,34)) # work as W2 (hidden_size, input_size)

def random_walk(vertex, t):
    ans = []
    ans.append(vertex)

    while True:
        # stop untill the window size get t
        if len(ans) == t:
            return ans

        # check nearest vertexs
        vertex_adj_list = list(adj[vertex])
        near_vertex_index = np.nonzero(vertex_adj_list)

        # choose the nearess vertex randomly
        get_vertex = np.random.choice(list(near_vertex_index[0]), 1)
        ans.append(get_vertex[0])
        vertex = get_vertex[0]


def skipgram(W, w, loss):
    new_loss = np.zeros(34)
    loss = 0
    for idx, vertex in enumerate(W):
        # making u_list considering w
        start = idx - w
        end = idx + w
        if start < 0:
            start = 0
        if end >= len(W):
            end = len(W) - 1

        u_list = []
        u_list.extend(W[start:idx])
        u_list.extend(W[idx + 1:end + 1])

        # calculating each u from u_list
        for each_u in u_list:
            # input, ouput with one-hot encoding
            input_vertex = np.zeros(34)
            y_pred = np.zeros(34)
            v = vertex
            u = each_u
            input_vertex[v] = 1
            y_pred[u] = 1

            # gradient (forward,backpropa)

            ERR, ERR2, diff, Y = n_network.gradient(input_vertex, y_pred)
            # updata params
            n_network.params['W1'] -= learning_rate * ERR
            n_network.params['W2'] -= learning_rate * ERR2
            # n_network.params['W2'] -= np.reshape(learning_rate * ERR2 * H.T, (d,34))

            # calculating loss
            loss += -np.log(Y[each_u])
        # new_loss = new_loss / len(u_list)
    return loss


epoch = 100
epoch_loss2 = []
loss = np.zeros((34, 34))
for _ in range(epoch):
    epoch_loss = 0
    for i in range(r):
        O = np.arange(34)
        np.random.shuffle(O)

        for vertex in O:
            W = random_walk(vertex, t)
            loss = skipgram(W, w, loss)
            epoch_loss += (loss / len(W))

    # h = np.dot(adj[1],W1)
    # H = softmax(U)

    # U2 = np.dot(h,W2)
    # Y = softmax(U2)
    # aa = np.mean(-np.log(Y -adj[1]))
    epoch_loss2.append(np.mean(epoch_loss) / (len(O) * r))
    # calculating loss
    # n_network.params['W1']


from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

plt.plot(epoch_loss2)
plt.title("the loss - the number of epochs")
plt.xlabel("Number of epochs")
plt.ylabel("loss")
plt.show()

nums = np.identity(34)
W1 = n_network.params['W1']
output = np.dot(nums , W1)
adj = nx.to_numpy_array(adjlist)
label = karate_label[:,-1]


node_number
label
label_fix = []
for i in node_number:
    tem = label[i]
    label_fix.append(tem)




import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

model = TSNE(learning_rate=100,perplexity=5)
transformed = model.fit_transform(output)
xs = transformed[:,0]
ys = transformed[:,1]

for i in range(len(xs)):
    plt.scatter(xs[i],ys[i],c = node_number[i])
    plt.text(xs[i],ys[i],i+1)
plt.scatter(xs,ys,c=label_fix)
#plt.text(xs,ys)
plt.show()