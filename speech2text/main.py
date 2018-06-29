import json
import scipy.io.wavfile as wav
import pandas
import argparse
from watson_developer_cloud import SpeechToTextV1
from deepspeech.model import Model
from timeit import default_timer as timer

class WatsonSTT:
    """ Wrap SpeechToTextV1 and provide the methods we need"""

    def __init__(self, config):
        self.config = config

    def connect(self):
        with open(self.config, 'r') as f:
            settings = json.load(f)
            self.conn = SpeechToTextV1(
                username=settings['username'],
                password=settings['password'],
                url=settings['url']
            )

    def oneshoot(self, wav_file):
        with open(wav_file, 'rb') as audio:
            start = timer()
            response = self.conn.recognize(
                audio=audio,
                content_type='audio/wav',
                timestamps=True,
                word_alternatives_threshold=0.9
            )
            latency = timer() - start
            print('watson STT costs %0.3fs.' % latency)

        return response, latency

    def get_sentences(self, json_result):
        rev = []
        if json_result is None:
            return rev

        sentence = ''
        for one in json_result['results']:
            sentence += one['alternatives'][0]['transcript']
        
        rev.append(sentence)
        return rev

class DeepSpeech:
    """Wrap DeepSpeech and provide the methods we need"""

    def __init__(self, config):

        self.beam_width = 1024
        self.lm_weight = 1.75
        self.word_count_weight = 1.00
        self.valid_word_count_weight = 1.00
        self.n_features = 26
        self.n_context = 9

        with open(config, 'r') as f:
            settings = json.load(f)
            self.alphabet = settings.get('alphabet')
            self.lm = settings.get('lm')
            self.trie = settings.get('trie')
            self.graph = settings.get('graph')

    def load_model(self):
        start = timer()
        self.model = Model(self.graph, self.n_features, self.n_context, self.alphabet, self.beam_width)
        end = timer()
        print('Loaded model in %0.3fs.' % (end - start))
        if self.lm is not None and self.trie is not None:
            start = timer()
            self.model.enableDecoderWithLM(
                self.alphabet, self.lm,
                self.trie, self.lm_weight,
                self.word_count_weight,
                self.valid_word_count_weight
            )
            end = timer()
            print('Loaded language model in %0.3fs.' % (end - start))

    def oneshoot(self, wav_file):
        fs, audio = wav.read(wav_file)
        start = timer()
        result = self.model.stt(audio, fs)
        latency = timer() - start
        audio_length = len(audio) * ( 1 / 16000)
        print('Inference took %0.3fs for %0.3fs audio file.' % (latency, audio_length))
        return result, latency

# The following code is from: http://hetland.org/coding/python/levenshtein.py
# This is a straightforward implementation of a well-known algorithm, and thus
# probably shouldn't be covered by copyright to begin with. But in case it is,
# the author (Magnus Lie Hetland) has, to the extent possible under law,
# dedicated all copyright and related and neighboring rights to this software
# to the public domain worldwide, by distributing it under the CC0 license,
# version 1.0. This software is distributed without any warranty. For more
# information, see <http://creativecommons.org/publicdomain/zero/1.0>

def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = list(range(n+1))
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def run(args):
    # Let's load the DeepSpeech model first
    ds = DeepSpeech(args.ds)
    ds.load_model()

    # Init Watson STT service
    stt = WatsonSTT(args.watson)
    stt.connect()
 
    output = []
    csv_data = pandas.read_csv(args.csv)
    total_watson_edit_distance = 0
    total_ds_edit_distance = 0

    for row in csv_data.itertuples():
        watson_results, watson_latency = stt.oneshoot(row[1])
        ds_results, ds_latency = ds.oneshoot(row[1])
        sentences = stt.get_sentences(watson_results)
        transcript = row[3].strip().lower()
        label = transcript.split()
    
        watson_res = sentences[0].strip().lower()
        watson_edit_distance = levenshtein(label, watson_res.split())

        ds_results = ds_results.strip()
        ds_edit_distance = levenshtein(label, ds_results.split())

        total_watson_edit_distance += watson_edit_distance
        total_ds_edit_distance += ds_edit_distance
        output.append({
            'wav_file': row[1],
            'transcript': transcript,
            'watson': watson_res,
            'watson_latency': watson_latency,
            'watson_distance': watson_edit_distance,
            'deepspeech': ds_results,
            'deepspeech_latency': ds_latency,
            'deepspeech_distance': ds_edit_distance
        })

        print("label     :%s" % transcript)
        print("watson    :%s" % watson_res)
        print("deepspeech:%s" % ds_results)
        print("levenshtein:%d (watson) vs %d (ds)" % (watson_edit_distance, ds_edit_distance))
        print("")

    print("avg levenshtein: %0.3f (watson) vs %0.3f (ds)" % (total_watson_edit_distance/csv_data.shape[0], total_ds_edit_distance/csv_data.shape[0]))

    pandas.DataFrame(output).to_csv('results.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='A tool to compare STT accuracy between Watson and DeepSpeech')
    parser.add_argument('--csv',
        help='Path to the csv file which contains the wav and label for STT')
    parser.add_argument('--watson',
        nargs='?',
        default='./watson.json',
        help='A JSON file which stores credentials for Watson Speech to Text service')
    parser.add_argument('--ds',
        nargs='?',
        default='./ds.json',
        help='A JSON file which stores settings for DeepSpeech')

    args = parser.parse_args()
    if args.csv is None:
        parser.print_usage()
        return

    run(args)

if __name__ == '__main__' :
    main()