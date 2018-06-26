'use strict';

const fs = require('fs');
const readline = require('readline');
const path = require('path');
const os = require('os');
const n2w = require('number-to-words');
const parse = require('csv-parse/lib/sync');
const settings = require('./config.json');

console.error(JSON.stringify(settings, null, 2));

// Check argv and print out usage if needed
function usage(argv) {
  if (!argv || argv.length !== 5) {
    console.info('usage: node . <origin csv> <data dir> <output prefix>');
    return false;
  }

  if (!fs.existsSync(argv[2])) {
    console.error(`csv: ${argv[2]} does not exist`);
    return false;
  }
  try {
    let stat = fs.lstatSync(argv[3]);
    if (!stat.isDirectory()) {
      console.error(`data directory: ${argv[3]} is not a directory`);
      return false;
    }
  } catch (e) {
    console.error(`data directory: ${argv[3]} does not exist`);
    return false;
  }
}

// Parse recording csv and get wav file and transcript and then generate
// csv file for DeepSpeech for each data directory
function parseCSV(originCSV, dataDir, output) {
  const rl = readline.createInterface({
    input: fs.createReadStream(originCSV),
    crlfDelay: Infinity
  });

  let csvData = {};
  let numRe = /\d+/;
  rl.on('line', (line) => {
    try {
      let columns = parse(line)[0];
      let transcript = columns[settings.transcript] ?
        columns[settings.transcript].toLowerCase() : '';
      transcript = transcript.replace(/[`",?!]/g, '')
        .replace(/-/g, ' ').replace(/\.$/, '').replace('km', 'kilometer');

      // Number to words
      let matched = numRe.exec(transcript);
      if (matched !== null) {
        transcript = transcript.replace(matched[0], n2w.toWords(matched[0]));
      }

      csvData[columns[settings.fileName]] = transcript.trim();
    } catch (e) {
      console.error(`can not parse this line:${line}`);
    }
  });

  rl.on('close', () => {
    parseDataDir(csvData, dataDir, output);
  });
}

// Read wav files from data directory and generate csv for DeepSpeech
function parseDataDir(csvData, data, output) {
  if (!path.isAbsolute(data)) {
    data = path.join(__dirname, data);
  }
  let files = fs.readdirSync(data);
  files.forEach((file) => {
    // Write a csv file which contains all wav files for this directory
    let fullPath = path.join(data, file);
    let stat = fs.lstatSync(fullPath);
    if (stat.isDirectory()) {
      // Get all wav files
      fs.readdir(fullPath, (error, files) => {
        let outputCSV = output;
        if (!path.isAbsolute(output)) {
          outputCSV = path.join(__dirname, output);
        }
        outputCSV = path.join(outputCSV, `${file}.csv`);
        let outputS = fs.createWriteStream(outputCSV);
        outputS.write(`wav_filename,wav_filesize,transcript\n`);
        files.forEach( (wav) => {
          // Write a line for each wav file
          let wavFull = path.join(fullPath, wav);
          let fSize = fs.lstatSync(wavFull);
          outputS.write(`${wavFull},${fSize.size},${csvData[wav]}\n`);
        });
        outputS.end();
      });
    }
  });
}


if (usage(process.argv) === false) {
  process.exit(1);
}

parseCSV(process.argv[2], process.argv[3], process.argv[4]);
