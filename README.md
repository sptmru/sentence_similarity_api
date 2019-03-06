# Sentence similarity API

### Description
This is a simple Flask-based API created for sentence similarity estimation (using Gensim).
The API has two endpoints: 
```/find_similar_sentences``` — accepts text, sentence and similarity estimation, returns sentences from text which are similar enough to the given sentence.
```/similarity_estimation``` — accepts two sentences and returns their similarity estimations