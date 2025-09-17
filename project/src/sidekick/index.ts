import ollama from 'ollama';

// iteration:
// get screenshot from memory folks (they expose a webserver or something)
// also get other info from memory like the last few actions the user has taken
// then we feed that into a small llm and have it predict the next action
// (later, we will incorporate additional factors like gaze position and such.)
// if most likely action is above a certain threshold, and second most likely is sufficiently lower, suggest/do the action
// example: spreadsheets, what else?
// Related: could we use a small LLM like this one to make easy decisions in agent loop, speeding it up?
// What do we take as input? Image of screen? OCR+SOM text?
// May want to make this a callback triggered by memory/recording system, or something
// So that we process screen changes as soon as they happen, and not when they're not happening

const message = { role: 'user', content: '...' };
const response = await ollama.chat({
  model: 'gemma3:4b',
  messages: [
    {
      "role": "user",
      "content": "Describe the image briefly.",
      "images": ["./ss3.png"],
    },
  ],
});

console.log(`${response.message.content}\n`);

const inputDuration = response.prompt_eval_duration / Math.pow(10, 9);
const outputDuration = response.eval_duration / Math.pow(10, 9);
const totalDuration = response.total_duration / Math.pow(10, 9);
const inputRate = response.prompt_eval_count / inputDuration;
const outputRate = response.eval_count / outputDuration;
console.log(`In:  ${response.prompt_eval_count} tokens in ${inputDuration.toFixed(2)}s (${inputRate.toFixed(2)} t/s)`);
console.log(`Out: ${response.eval_count} tokens in ${outputDuration.toFixed(2)}s (${outputRate.toFixed(2)} t/s)`);
console.log(`Total duration: ${totalDuration.toFixed(2)}s`);
