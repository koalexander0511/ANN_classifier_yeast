#  1.  Sequence Name: Accession number for the SWISS-PROT database
#  2.  mcg: McGeoch's method for signal sequence recognition.
#  3.  gvh: von Heijne's method for signal sequence recognition.
#  4.  alm: Score of the ALOM membrane spanning region prediction program.
#  5.  mit: Score of discriminant analysis of the amino acid content of
#       the N-terminal region (20 residues long) of mitochondrial and 
#           non-mitochondrial proteins.
#  6.  erl: Presence of "HDEL" substring (thought to act as a signal for
#       retention in the endoplasmic reticulum lumen). Binary attribute.
#  7.  pox: Peroxisomal targeting signal in the C-terminus.
#  8.  vac: Score of discriminant analysis of the amino acid content of
#           vacuolar and extracellular proteins.
#  9.  nuc: Score of discriminant analysis of nuclear localization signals
#       of nuclear and non-nuclear proteins.

import numpy as np
import matplotlib.pyplot as plt
from numpy import exp
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import tensorflow as tf

#input and output values with outliers
raw_x=[]
raw_y=[]

#input and out values without outliers
x=[]
y=[]

f=open("yeast.data","r")
for i in f:
    d = i.split()
    raw_x.append(list(map(float,d[1:9])))
    if(d[9] == "CYT"):
        raw_y.append([1,0,0,0,0,0,0,0,0])
    elif(d[9] == "NUC"):
        raw_y.append([0,1,0,0,0,0,0,0,0])
    elif(d[9] == "MIT"):
        raw_y.append([0,0,1,0,0,0,0,0,0])
    elif(d[9] == "ME3"):
        raw_y.append([0,0,0,1,0,0,0,0,0])
    elif(d[9] == "ME2"):
        raw_y.append([0,0,0,0,1,0,0,0,0])
    elif(d[9] == "ME1"):
        raw_y.append([0,0,0,0,0,1,0,0,0])
    elif(d[9] == "EXC"):
        raw_y.append([0,0,0,0,0,0,1,0,0])
    elif(d[9] == "VAC"):
        raw_y.append([0,0,0,0,0,0,0,1,0])
    elif(d[9] == "POX"):
        raw_y.append([0,0,0,0,0,0,0,0,1])
    else:
        raw_y.append([0,0,0,0,0,0,0,0,0])
f.close()


print(   "\n################"   )
print(   "#Problem 1"         )
print(   "################\n"   )



#Outlier detection and removal

isof = IsolationForest()
lof=LocalOutlierFactor(n_neighbors=300)    
x_lof_pred=lof.fit_predict(raw_x)
x_isof_pred=isof.fit_predict(raw_x)

print("Number of data flagged as outliers using LOF: ", sum(x_lof_pred==-1))
print("Number of data flagged as outliers using Isolation Forest: ",sum(x_isof_pred==-1))
print("Number of data flagged differently between two methods: ", sum(x_lof_pred != x_isof_pred))


#Create data w/out outliers
for i in range(0,len(raw_x)):
    if(x_lof_pred[i] == 1):
        x.append(raw_x[i])
        y.append(raw_y[i])




print(   "\n################"   )
print(   "#Problem 2"         )
print(   "################\n"   )



epch = 5000 #epoch size parameter
batch = 100 #batch size parameter
learning_rate = 1 #learning rate for the gradient descent


#split data into training and testing set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 42)

#construct NN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(3, input_shape=(8,), activation='sigmoid'))
model.add(tf.keras.layers.Dense(3, activation='sigmoid'))
model.add(tf.keras.layers.Dense(9, activation='softmax'))

sgd = tf.keras.optimizers.SGD(lr = learning_rate)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#remember initial weights so we can reset the model if we choose to
InitialWeights = model.get_weights()


#containers for plotting data
w11_L3 = []     #weight on layer 3, coming from node 1(layer 2) to node 1(layer 3)
w12_L3 = []     #weight on layer 3, coming from node 2(layer 2) to node 1(layer 3)
w13_L3 = []     #weight on layer 3, coming from node 3(layer 2) to node 1(layer 3)
b_L3 = []       #bias weight on layer 3, coming from bias node(layer 2) to node 1(layer 3)

train_losses = []
val_losses = []
train_accuracy = [] 
val_accuracy = []




#loop on one epoch in order to record data
for i in range(0,epch):
    history = model.fit(x_train,y_train,epochs=1, batch_size=batch, verbose=2)
    
    train_accuracy.append(history.history['acc'])
    val_loss, val_acc = model.evaluate(x_test,y_test)
    val_accuracy.append(val_acc)
    
    #get weights from layer 3(last layer)
    weights = model.layers[2].get_weights()
    
    w11_L3.append(weights[0][0][0])
    w12_L3.append(weights[0][1][0])
    w13_L3.append(weights[0][2][0])
    b_L3.append(weights[1][0])

OptimalWeights = model.get_weights()

#plotting training and validation accuracy
plt.plot(train_accuracy, label = "Training Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.legend()
plt.title("Validation and Training Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Iterations")
plt.show()

#plotting weight values regarding "CYT" node
plt.plot(w11_L3, label="W11")
plt.plot(w12_L3, label="W12")
plt.plot(w13_L3, label="W13")
plt.plot(b_L3, label="Bias")
plt.legend()
plt.title("Weights at the last layer from CYT")
plt.ylabel("Weight Values")
plt.xlabel("Iterations")
plt.show()




print(   "\n################"   )
print(   "#Problem 3"         )
print(   "################\n"   )



model.set_weights(InitialWeights)   #Reset the model

history = model.fit(raw_x,raw_y,epochs=epch,batch_size=batch, verbose=0)
print("Training Error: ", 1-history.history['acc'][len(history.history['acc'])-1])

print("Weights matrix after training on whole dataset: ", model.get_weights())



print(   "\n################"   )
print(   "#Problem 4"         )
print(   "################\n"   )



L1_WeightArray = [np.array(
                    [
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0],
                        [0,0,0]
                     ]),
                    np.array([0,0,0])
                 ]

L2_WeightArray = [np.array(
                    [
                        [1,0,0],
                        [1,0,0],
                        [1,0,0]
                    ]),
                    np.array([1,0,0])
                ]

L3_WeightArray = [np.array(
                    [
                        [1,0,0,0,0,0,0,0,0],
                        [1,0,0,0,0,0,0,0,0],
                        [1,0,0,0,0,0,0,0,0]
                    ]),
                    np.array([1,0,0,0,0,0,0,0,0])
                ]


model.layers[0].set_weights(L1_WeightArray)
model.layers[1].set_weights(L2_WeightArray)
model.layers[2].set_weights(L3_WeightArray)



print("Training data x:", x_train[0])
print("Training data y:", y_train[0])

weights = model.layers[2].get_weights()
print("Initial weights:")
print(weights[0])

print("Output node values with initial weights:",model.predict(np.array([x_train[0]])))

history = model.fit(np.array([x_train[0]]),np.array([y_train[0]]),ephochs=1)
weights_L3 = model.layers[2].get_weights()
print("Layer 3:")
print("W11: ", weights_L3[0][0][0])
print("W12: ", weights_L3[0][1][0])
print("W13: ", weights_L3[0][2][0])
print("W14(Bias): ", weights_L3[1][0])

weights_L2 = model.layers[1].get_weights()
print("Layer 2:")
print("W11: ", weights_L2[0][0][0])
print("W12: ", weights_L2[0][1][0])
print("W13: ", weights_L2[0][2][0])
print("W14(Bias): ", weights_L2[1][0])


print("Weights matrix of the total model:")
print(model.get_weights())

print(   "\n################"   )
print(   "#Problem 5"         )
print(   "################\n"   )



def flexModel(numLayers,numNodes):
    print("evaluating model with ", numLayers, " hidden layers, with ", numNodes, " nodes each")
    model2 = tf.keras.Sequential()
    model2.add(tf.keras.layers.Dense(numNodes, input_shape=(8,), activation='sigmoid'))
    for i in range(0,numLayers-1):
        model2.add(tf.keras.layers.Dense(numNodes, activation='sigmoid'))
    model2.add(tf.keras.layers.Dense(9, activation='softmax'))
    
    sgd = tf.keras.optimizers.SGD(lr=learning_rate)
    
    model2.compile(optimizer=sgd,
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    
    model2.fit(x_train,y_train,epochs=epch, batch_size=32, verbose=0)
    
    ls, accurecy = model2.evaluate(x_test,y_test)
    
    return 1-accurecy


errorMatrix = []
tmp = []

for i in range(1,4):
    tmp.clear()
    for j in range(1,5):
        tmp.append(flexModel(i,j*3))
    errorMatrix.append(tmp)

print("Error Matrix for Different model configurations: ", errorMatrix)




print(   "\n################"   )
print(   "#Problem 6"         )
print(   "################\n"   )



model.set_weights(OptimalWeights)

result = model.predict(np.array([[0.52,0.47,0.52,0.23,0.55,0.03,0.52,0.39]]))
maxval = max(result[0])

if(result[0][0] == maxval):
    print("CYT")
elif(result[0][1] == maxval):
    print("NUC")
elif(result[0][2] == maxval):
    print("MIT")
elif(result[0][3] == maxval):
    print("ME3")
elif(result[0][4] == maxval):
    print("ME2")
elif(result[0][5] == maxval):
    print("ME1")
elif(result[0][6] == maxval):
    print("EXC")
elif(result[0][7] == maxval):
    print("VAC")
elif(result[0][8] == maxval):
    print("POX")
