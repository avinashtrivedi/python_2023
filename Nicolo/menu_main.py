import pandas as pd
import json
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MEModel(tf.keras.Model):
    def __init__(self, filename):
        with open(filename, "r") as f:
            items = [ json.loads(line) for line in f ]
            
        self.df = pd.DataFrame({
            "id": [x["_source"]["id"] for x in items],
            "name": [x["_source"]["name"] for x in items],
            "entype": [x["_source"]["entype"] for x in items],
        })

        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

        self.df['Embeddings'] = self.df['name'].apply(lambda x: self.model.encode(x))
        
        
    def get_top5_embedding_rows(self, user_input):
        # convert user_input to embeddings
        vect2 = self.model.encode(user_input).reshape(1,-1)

        # make a copy of original dataframe
        df_temp = self.df.copy()

        df_temp['similarity'] = df_temp['Embeddings'].apply(lambda vect1:cosine_similarity(vect1.reshape(1,-1), vect2)[0][0])

        return df_temp.sort_values('similarity',ascending=False).head(10)
                
    
    def call(self, inputs, training=None, mask=None):
        return self.get_top5_embedding_rows(inputs)


me_model = MEModel("msample.json")

me_model.ge

tf.saved_model.save(me_model, 'model')

