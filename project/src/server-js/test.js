import fs from 'fs';
import Speaker from '@mastra/node-speaker';

const speaker = new Speaker({
  channels: 1,
  bitDepth: 16,
  sampleRate: 24000,
  signed: true,
});

const base64 = fs.readFileSync('./a.b64', 'utf8');
const pcm = Buffer.from(base64, 'base64');
// fs.writeFileSync('b.pcm', pcm);
// const pcm = fs.readFileSync('./a.pcm');
speaker.write(pcm);
setTimeout(() => {
  console.log('...done');
}, 10_000);
