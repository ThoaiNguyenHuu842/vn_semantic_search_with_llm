from sentence_transformers import SentenceTransformer, util
import torch
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

corpus = ['Rolling is a piano store',
          'Financial instrument is used in finance to describe contracts',
          ]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

query = 'Where to buy musical instrument?'

query_embedding = model.encode(query, convert_to_tensor=True)
cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
top_result = torch.topk(cos_scores, k=1)

print('Query sentence:')
print(query)
print('Closest match:')
print(corpus[top_result.indices[0]])
