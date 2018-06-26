# DeepSpeech CSV Composer
An utility tool to generate the CSV for [DeepSpeech](https://github.com/mozilla/DeepSpeech).

### Install the dependencies
```
npm install
```

### Configuration
The utility reads an CSV as input, which contains the wav filename and transcript and maybe other information in each row. In the `config.json`, you need to specify the column index (starting from 0) for both wav file name and transcript.

The example content of `config.json`
```JSON
{
  "fileName": 9,
  "transcript": 10
}
```
`fileName` denotes the column index for the file name of wav file.
The file path could be absolute path or relative path to the `index.js`.

`transcript` denotes the column index for the transcript.

### Usage
In order to run the utility, you need to put your data set into 3 directories: `train`, `test` and `validate`. And make sure the file name
in the `fileName` column point to the wav files inside these directories. And use the following command to generate the CSV for DeepSpeech:
```
node index.js <your CSV> <parent directory of train, test and validate> <csv output directory>
``` 