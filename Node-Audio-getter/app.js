const express = require('express');
const multer = require('multer');
const fs = require('fs');
const { spawn } = require('child_process');
const path = require('path');
const axios = require('axios');

const app = express();
const upload = multer();
const uploadDir = path.join(__dirname, 'uploads');

// Create uploads directory if it doesn't exist
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

function runPythonScript(scriptPath, audioFilePath) {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [scriptPath, audioFilePath]);

    let scriptOutput = "";
    pythonProcess.stdout.on('data', (data) => {
      scriptOutput += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      reject(data.toString());
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(`Python script exited with code ${code}`);
      } else {
        resolve(scriptOutput);
      }
    });
  });
}

app.post('/upload', upload.single('audioFile'), (req, res) => {
  if (!req.file) {
    return res.status(400).send('No audio file uploaded.');
  }

  // Save the audio file to the upload directory
  const filePath = path.join(uploadDir, 'audio.mp3');
  fs.writeFile(filePath, req.file.buffer, (err) => {
    if (err) {
      console.error('Error saving audio file:', err);
      return res.status(500).send('Error saving audio file.');
    }

    console.log('Audio file saved:', filePath);

    // Run Python script
    const scriptPath = 'insanely-fast-whisper.py';
    const audioFilePath = path.relative(__dirname, filePath); // Get relative path

    runPythonScript(scriptPath, audioFilePath)
      .then(output => {
        console.log('Python script output:', output);
        res.status(200).send(output);
      })
      .catch(error => {
        console.error('Error running Python script:', error);
        res.status(500).send('Error running Python script.');
      });
  });
});

app.listen(9020, () => {
  console.log('Server is running on port 9020');

  // Send uploadFile directory to Python API after 2 minutes
  setTimeout(() => {
    const postData = {
      path: 'C:/Users/katoc/Downloads/node-audio-getter/uploadFile'
    };
    axios.post('http://localhost:8000/store_data', postData, {
      headers: {
        'Content-Type': 'application/json'
      }
    }).then(response => {
      console.log('Axios POST request successful:', response.data);
    }).catch(error => {
      console.error('Error sending Axios POST request:', error);
    });
  }, 120000); // 2 minutes delay (120000 milliseconds)
});
