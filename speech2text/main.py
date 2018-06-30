import json
import scipy.io.wavfile as wav
import pandas
import argparse
import speech_recognition as sr
import re
from watson_developer_cloud import SpeechToTextV1
from deepspeech.model import Model
from timeit import default_timer as timer

class STTServices:
    """ Wrap SpeechToTextV1 and provide the methods we need"""

    def __init__(self, settings):
        self.settings = {}
        for key in settings:
            self.settings[str(key).lower()] = settings[key]

    def init(self):
        self.recognizer = sr.Recognizer()

    def oneshoot(self, wav_file):
        with sr.AudioFile(wav_file) as source:
            audio = self.recognizer.record(source)
        results = {}

        for key in self.settings:
            start = timer()
            if key == 'google':
                results[key] = { 'res': self.recognizer.recognize_google_cloud(audio, credentials_json=self.settings[key]['credentials'])}
            elif key == 'watson':
                results['watson'] = { 'res': self.recognizer.recognize_ibm(audio, 
                    username=self.settings[key]['username'], password=self.settings[key]['password'])}
            elif key == 'bling':
                results['bling'] = {'res': self.recognizer.recognize_bing(audio, key=self.settings[key]['key'])}

            results[key]['latency'] = timer() - start

        return results

class DeepSpeech:
    """Wrap DeepSpeech and provide the methods we need"""

    def __init__(self, settings):

        self.beam_width = 1024
        self.lm_weight = 1.75
        self.word_count_weight = 1.00
        self.valid_word_count_weight = 1.00
        self.n_features = 26
        self.n_context = 9
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
    with open(args.config, 'r') as f:
        settings = json.load(f)

    # Let's load the DeepSpeech model first
    ds = DeepSpeech(settings.pop('deepspeech'))
    ds.load_model()

    # Init Watson STT service
    stt = STTServices(settings)
    stt.init()
 
    outputs = []
    csv_data = pandas.read_csv(args.csv)
    total_distance = {'deepspeech':0}
    for key in stt.settings:
        total_distance[key] = 0

    end_period = re.compile("\.$")

    for row in csv_data.itertuples():
        results = stt.oneshoot(row[1])
        ds_results, ds_latency = ds.oneshoot(row[1])
        transcript = row[3].strip().lower()
        label = transcript.split()
    
        output = {'wav_file': row[1], 'transcript': transcript, 
                'deepspeech': ds_results.strip(), 'deepspeech_latency': ds_latency,
                'deepspeech_distance': levenshtein(label, ds_results.split())}
        total_distance['deepspeech'] += output['deepspeech_distance']

        for result in results:
            values = results[result]
            res = end_period.sub('', values['res'].strip().lower())
            output[result] = res
            output[result + '_latency'] = values['latency']
            output[result + '_distance'] = levenshtein(label, res.split())
            total_distance[result] += output[result + '_distance']

        print(json.dumps(output, indent=2))
        print('')
        total_distance['deepspeech'] += output['deepspeech_distance']
        outputs.append(output)

    print("avg levenshtein:")
    print(json.dumps(total_distance, indent=2))

    pandas.DataFrame(outputs).to_csv('results.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='A tool to compare STT accuracy between Watson and DeepSpeech')
    parser.add_argument('--csv',
        help='Path to the csv file which contains the wav and label for STT')
    parser.add_argument('--config',
        nargs='?',
        default='./config.json',
        help='A JSON file which stores credentials for multiple Speech to Text Service')

    args = parser.parse_args()
    if args.csv is None:
        parser.print_usage()
        return

    run(args)

if __name__ == '__main__' :
    main()