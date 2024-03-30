from sentence_transformers import SentenceTransformer, util
import torch
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

corpus = ['Bài hát Tết Đến Rồi',
          'Năm nay nông dân được mùa vụ xuân hè',
          ]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

query = 'Ca khúc về mùa xuân'

query_embedding = model.encode(query, convert_to_tensor=True)
cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
top_result = torch.topk(cos_scores, k=1)

print('Query sentence:')
print(query)
print('Closest match:')
print(corpus[top_result.indices[0]])
