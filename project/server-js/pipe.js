import fs from 'fs';
import OpenAI from 'openai';

const filename = 'sample3.wav';

// const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
// const transcription = await groq.audio.transcriptions.create({
//   file: fs.createReadStream(filename),
//   model: 'whisper-large-v3-turbo',
// });
// console.log(transcription.text);

// const openai = new OpenAI();
// const transcription = await openai.audio.transcriptions.create({
//   file: fs.createReadStream(filename),
//   model: 'gpt-4o-mini-transcribe',
// });
// console.log(transcription.text);

const gemini = new OpenAI({ baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/", apiKey: process.env.GEMINI_API_KEY });
const audioFile = fs.readFileSync(filename);
const base64Audio = Buffer.from(audioFile).toString('base64');
const response = await gemini.chat.completions.create({
  model: 'gemini-2.5-flash',
  messages: [
    {
      role: 'user',
      content: [
        {
          type: 'text',
          text: 'Transcribe the audio.',
        },
        {
          type: 'input_audio',
          input_audio: {
            data: base64Audio,
            format: 'wav',
          },
        },
      ],
    },
  ],
  reasoning_effort: 'none',
});
console.log(response.choices[0].message.content);
