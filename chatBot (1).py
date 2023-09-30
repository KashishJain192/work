#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle 
import numpy as np 


# In[2]:


# importing the data 
with open('train_qa220120145526-220818-175522.txt' , "rb") as fp:
    train_data = pickle.load(fp)


# In[3]:


len(train_data)


# In[4]:


with open('test_qa220120145430-220818-175426 (1).txt' , "rb") as fp:
    test_data = pickle.load(fp)


# In[5]:


len(test_data)


# In[6]:


# creating vocabulary set 
vocab = set()


# In[7]:


all_data = train_data+test_data


# In[8]:


# adding stories and questions to vocabulary
for story,question,answer in all_data:
    vocab=vocab.union(set(story))
    vocab=vocab.union(set(question))
    


# In[9]:


#adding answers to vocab 
vocab.add('yes')
vocab.add('no')


# In[10]:


#checking the length of vocab
len(vocab)


# In[11]:


max_story_len = max([len(data[0]) for data in all_data])


# In[12]:


max_story_len


# In[13]:


max_question_len = max([len(data[1]) for data in all_data])


# In[14]:


max_question_len


# In[15]:


pip install keras 


# In[16]:


from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[17]:


from keras.preprocessing.text import Tokenizer


# In[18]:


tokenizer = Tokenizer(filters = [])


# In[19]:


tokenizer.fit_on_texts(vocab)


# In[20]:


tokenizer.word_index


# In[21]:


# creating training dataset

train_story_texts = []
train_question_texts =[]
train_answers =[]

for story,question,answers in train_data :
    train_story_texts.append(story)
    train_question_texts.append(question)
    


# In[22]:


train_story_seq = tokenizer.texts_to_sequences(train_story_texts)


# In[36]:


vocab_len=len(vocab)+1


# In[37]:


def vectorize_stories(data,word_index=tokenizer.word_index,
                     max_story_len=max_story_len,
                     max_question_len=max_question_len):
    X=[]
    XQ=[]
    Y=[]
    
    for story,question,answer in data:
        x=[word_index[word.lower()]for word in story]
        xq=[word_index[word.lower()]for word in question]
        y=np.zeros(len(word_index)+1)
        y[word_index[answer]]=1
        
        
    X.append(x)
    XQ.append(xq)
    Y.append(y)
    
    
    return(pad_sequences(X,maxlen=max_story_len),
          pad_sequences(XQ,maxlen=max_question_len),
          np.array(Y))
    


# In[38]:


input_train, queries_train,answers_train =vectorize_stories(train_data)


# In[39]:


input_test, queries_test,answers_test =vectorize_stories(test_data)


# In[40]:


from keras.models import Sequential,Model
from tensorflow.keras.layers import Embedding
from keras.layers import Input,Activation,Dense,Permute,Dropout,add,dot,concatenate,LSTM


# In[41]:


input_sequence = Input((max_story_len))
question=Input((max_question_len))


# In[42]:


#input emcoder M
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_len,output_dim=64))
input_encoder_m.add(Dropout(0.3))


# In[43]:


#input emcoder c
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_len,output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))


# In[44]:


#input emcoder M
#input_encoder_c = Sequential()
#input_encoder_c.add(Embedding(input_dim=vocab_len,output_dim=64))
#input_encoder_m.add(Dropout(0.3))


# In[45]:


#question emcoder M
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_len,output_dim=64,input_length=max_question_len))
question_encoder.add(Dropout(0.3))


# In[46]:


input_encoded_m=input_encoder_m(input_sequence)
input_encoded_c=input_encoder_c(input_sequence)
question_encoded=question_encoder(question)


# In[47]:


match=dot([input_encoded_m,question_encoded],axes=(2,2))
match=Activation('Softmax')(match)


# In[48]:


response=add([match,input_encoded_c])
response=Permute((2,1))(response)


# In[49]:


answer = concatenate([response,question_encoded])


# In[50]:


answer=LSTM(32)(answer)


# In[51]:


answer=Dropout(0.5)(answer)


# In[52]:


answer=Dense(vocab_len)(answer)


# In[53]:


answer=Activation('Softmax')(answer)


# In[54]:


model=Model([input_sequence,question],answer)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


# 

# In[55]:


model.summary()


# In[43]:


history=model.fit([input_train,queries_train],answers_train,
                  batch_size=30,epochs=22,
                 validation_data=([input_test,queries_test],answers_test))


# In[44]:


import matplotlib.pyplot as plt 
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")


# In[45]:


#save
model.save("chatbot_model")


# In[46]:


#evaluation on test set
model.load_weights("chatbot_model")


# In[47]:


pred_results=model.predict(([input_test,queries_test]))


# In[48]:


pred_results


# In[49]:


test_data[0][0]


# In[50]:


story=' '.join(word for word in test_data[100][0])


# In[51]:


story


# In[52]:


query=' '.join(word for word in test_data[100][1])


# In[53]:


query


# In[54]:


test_data[100][2]


# In[55]:


val_max=np.argmax(pred_results[0])
for key ,val in tokenizer.word_index.items():
    if val == val_max:
      k = key 
print("Predicted answer is ",k)
print("Probability of certainity ",pred_results[0][val_max])


# In[ ]:




