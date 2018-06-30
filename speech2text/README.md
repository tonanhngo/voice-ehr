# Utility to compare multiple Speech to Text service and DeepSpeech
This tool reads input from a CSV file which contains wav_file and transcript columns.
Then it sends the wav file to multiple STT services and DeepSpeech. The results are
stored into `results.csv`

### Install the dependencies
```
pip3 install deepspeech*.whl
pip3 install -r requirement
```
This tools needs the deepspeech 0.2. We provide the wheel file for linux.
You also need cuda 9 and cudnn 7 in your system.

### Configuration
It needs one configuration file - `config.json` which contains credentials of multiple
Speech to Text services:

An example of config.json:
```
{
  "watson": {
    "url": "https://stream.watsonplatform.net/speech-to-text/api",
    "username": "XXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXX",
    "password": "XXXXXXXXXX"
  },
  "bling": {
    "key": "xxxxxxxxxxxxx"
  },
  "deepspeech": {
    "graph": "absolute path to frozen graph",
    "alphabet": "absolute path to alphabet file"
  }
}
```

### Usage
You need to prepare a CSV file with and contains at least `wav_file` and `transcript` columns.
`wav_file` is the absolute path to the wav file. and `transcript` is the transcript.

Then use the following command to perform the comparison:
```
python3 main.py --csv csv_file --config config.json
``` 
