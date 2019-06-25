# An implementation of Sequence to Sequence learning for performing addition.

from __future__ import print_function
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np
from six.moves import range

class CharacterTable(object):
    """Given a set of characters:
        +Encode them to one-hot integer representation
        +Decode the one-hot or integer representation to their character output
        +Decode a vector of probabilities to their character output
    """
    
    def __init__(self,chars):
        """Initialize the character table.
        
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars)) # gets the set of characters without duplicities and then sorts them
        # make dictionary of character and index: the final index of a duplicate character will be used
        self.char_indices = dict((c,i) for i,c in enumerate(self.chars))
        # make dictionary of index and character: every character gets an index, even if duplicate
        self.indices_char = dict((i,c) for i,c in enumerate(self.chars))
        
    def encode(self,C,num_rows):
        """One-hot encoding given string C.
        
        # Arguments
            C: string to be encoded
            num_rows: Number of rows in the returned one-hot encoding. This is used to keep the # of rows for each
            data the same.
        """
        x = np.zeros((num_rows,len(self.chars)))
        for i,c in enumerate(C):
            x[i,self.char_indices[c]] = 1
        return x
    
    def decode(self,x,calc_argmax = True):
        """Decode the given vector or 2D array to their character output.
        
        # Arguments
            x: A vector or 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with 'calc_argmax=False').
            calc_argmax = Whether to find character index with maximum probability, 
                defaults to True.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'
    
# Parameters for the model and data set. 
TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = True

# Maximum length of input is 'int + int' (eg. '345+678'). Maximum length of int is DIGITS.
MAXLEN = DIGITS + 1 + DIGITS

# All the numbers, plus sign and space for padding.
chars = '0123456789+ '
ctable = CharacterTable(chars)

questions = []
expected = []
seen = set()
print('Generating data...')
# following function gives a random integer to questions character
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789'))
                    for i in range(np.random.randint(1,DIGITS+1))))
    a,b = f(), f()
    
    # Skip any addition questions we haven't seen
    # Also skip any x+Y == Y+x (hence the sorting).
    key = tuple(sorted((a,b)))
    if key in seen:
        continue
    seen.add(key)
    # Pad the data with spaces such that it is always MAXLEN
    q = '{}+{}'.format(a,b)
    query = q + ''*(MAXLEN - len(q))
    ans = str(a+b)
    # Answers can be of maximum size DIGITS + 1 
    ans += '' * (DIGITS + 1 - len(ans))
    
    if REVERSE:
        # Reverse the query, eg.'12+345 ' becomes ' 543+21'.(Note the space used for padding.)
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions: ',len(questions))

print('Vectorization...')
x = np.zeros((len(questions),MAXLEN,len(chars)),dtype=np.bool)
y = np.zeros((len(questions),DIGITS+1,len(chars)),dtype = np.bool)
for i,sentence in enumerate(questions):
    x[i] = ctable.encode(sentence,MAXLEN)
for i,sentence in enumerate(expected):
    y[i] = ctable.encode(sentence,DIGITS+1)

# Shuffle (x,y) in unison as the later parts of x will almost all be larger digits.
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% validation data that we never train over.
split_at = len(x) - len(x)//10
(x_train,x_val) = x[:split_at],x[split_at:]
(y_train,y_val) = y[:split_at],y[split_at:]

print('Training data:')
print(x_train.shape)
print(y_train.shape)

print('Validation data:')
print(x_val.shape)
print(y_val.shape)

# TRY REPLACING GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 128 
BATCH_SIZE = 128
LAYERS = 1 

print('Build model...')
model = Sequential()
# Encode the input sequence using a RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length, use 
# input_shape = (None, num_feature).
model.add(RNN(HIDDEN_SIZE,input_shape=(MAXLEN,len(chars))))  
# As the decoder RNN's input, repetedly provide with the last output of 
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum 
# length of output. eg when DIGITS = 3, max output is 999 + 999 = 1998.
model.add(layers.RepeatVector(DIGITS+1))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    # By setting the return_sequences to True, return not only the last output but 
    # all the outputs so far in the form of (num_samples, timesteps, output_dim).
    # This is necesary as TimeDistributed in the below expects the first dimension
    # to be the timesteps. 
    model.add(RNN(HIDDEN_SIZE,return_sequences = True))

# Apply a dense layer to every temporal slice of an input. For each of the step of the 
# the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(len(chars),activation  ='softmax')))
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# Train the model each generation and show predictions against the validation data set
for iteration in range(1,200):
    print()
    print('-'*50)
    print('Iteration',iteration)
    model.fit(x_train,y_train,
              batch_size = BATCH_SIZE,
              epochs = 1,
              validation_data = (x_val,y_val))
    # SElect 10 samples from the validation at random so that we can visualize errors.
    for i in range(10):
        ind = np.random.randint(0,len(x_val))
        rowx,rowy = x_val[np.array([ind])],y_val[np.array([ind])]
        preds = model.predict_classes(rowx,verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0],calc_argmax=False)
        print('Q',q[::-1] if REVERSE else q, end=' ')
        print('T',correct,end = ' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)
    
    
    
    
    
    
    
    
    
    
    