import re
import gensim.downloader as gensim_api
from gensim.matutils import softcossim
from gensim import corpora
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
from werkzeug.contrib.fixers import ProxyFix


parser = reqparse.RequestParser()

app = Flask(__name__)
api = Api(app)

word2vec_model300 = gensim_api.load('word2vec-google-news-300')


def compare_sentences(sentence1, sentence2, model=word2vec_model300):
    sentence1 = sentence1.split()
    sentence2 = sentence2.split()

    documents = [sentence1, sentence2]
    dictionary = corpora.Dictionary(documents)
    ws1 = dictionary.doc2bow(sentence1)
    ws2 = dictionary.doc2bow(sentence2)

    similarity_matrix = model.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0,
                                                nonzero_limit=100)
    return softcossim(ws1, ws2, similarity_matrix)


class SimilarityEstimationEndpoint(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        try:
            sentence1 = json_data['sentence1']
            sentence2 = json_data['sentence2']
        except KeyError:
            return jsonify({'error': 'Please make sure both sentences are included in the request'})

        similarity_estimation = compare_sentences(sentence1, sentence2)
        result = {'similarity_estimation': similarity_estimation}

        return jsonify(result)


class FindingSimilarSentencesEndpoint(Resource):
    def post(self):
        result = []
        json_data = request.get_json(force=True)
        try:
            text = json_data['text']
            sentence = json_data['sentence']
            similarity_estimation = json_data['similarity_estimation']
        except KeyError:
            return jsonify({'error': 'Please make sure that text, sentence and similarity_estimation '
                                     'are included in the request'})

        splitted_text = re.split(r"\.\.\.|[\n.:!?]", text)
        for text_sentence in splitted_text:
            se = compare_sentences(text_sentence, sentence)
            if text_sentence != '' and se >= similarity_estimation:
                sentence_result = {}
                sentence_result['sentence'] = text_sentence.strip()
                sentence_result['similarity_estimation'] = se
                result.append(sentence_result)
        return jsonify(result)


@app.errorhandler(404)
def page_not_found(err):
    return "page not found"


@app.errorhandler(500)
def raise_error(error):
    return error


api.add_resource(FindingSimilarSentencesEndpoint, '/find_similar_sentences')
api.add_resource(SimilarityEstimationEndpoint, '/similarity_estimation')


app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == '__main__':
    app.run()
