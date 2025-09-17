#!/usr/bin/env python3

import json
import sys
import argparse

def extract_token_counts(file_path):
    """Extract token counts from JSON log file."""
    input_audio_tokens_total = 0
    input_text_tokens_total = 0
    input_audio_tokens_cached = 0
    input_text_tokens_cached = 0
    output_audio_tokens = 0
    output_text_tokens = 0

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or 'audio_tokens' not in line:
                    continue

                try:
                    data = json.loads(line)
                    usage = data.get('response', {}).get('usage', {})

                    # Input token details
                    input_details = usage.get('input_token_details', {})
                    input_audio_tokens_total += input_details.get('audio_tokens', 0)
                    input_text_tokens_total += input_details.get('text_tokens', 0)

                    # Cached token details
                    cached_details = input_details.get('cached_tokens_details', {})
                    input_audio_tokens_cached += cached_details.get('audio_tokens', 0)
                    input_text_tokens_cached += cached_details.get('text_tokens', 0)

                    # Output token details
                    output_details = usage.get('output_token_details', {})
                    output_audio_tokens += output_details.get('audio_tokens', 0)
                    output_text_tokens += output_details.get('text_tokens', 0)

                except json.JSONDecodeError:
                    continue

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)

    # Calculate non-cached tokens
    input_audio_tokens = input_audio_tokens_total - input_audio_tokens_cached
    input_text_tokens = input_text_tokens_total - input_text_tokens_cached

    return {
        'input_audio_tokens': input_audio_tokens,
        'input_text_tokens': input_text_tokens,
        'input_audio_tokens_cached': input_audio_tokens_cached,
        'input_text_tokens_cached': input_text_tokens_cached,
        'output_audio_tokens': output_audio_tokens,
        'output_text_tokens': output_text_tokens
    }

def calculate_costs(tokens):
    """Calculate costs for GPT-4 Realtime and GPT-4 Mini Realtime."""

    # GPT-4 Realtime pricing (per million tokens)
    gpt_rt_audio_input_cost = 32.00
    gpt_rt_cached_audio_input_cost = 0.40
    gpt_rt_audio_output_cost = 64.00
    gpt_rt_text_input_cost = 4.00
    gpt_rt_cached_text_input_cost = 0.40
    gpt_rt_text_output_cost = 16.00

    # GPT-4 Mini Realtime pricing (per million tokens)
    gpt_mini_rt_audio_input_cost = 10.00
    gpt_mini_rt_cached_audio_input_cost = 0.30
    gpt_mini_rt_audio_output_cost = 20.00
    gpt_mini_rt_text_input_cost = 0.60
    gpt_mini_rt_cached_text_input_cost = 0.30
    gpt_mini_rt_text_output_cost = 2.40

    # Calculate GPT-4 Realtime costs
    gpt_rt_costs = {
        'audio_input': tokens['input_audio_tokens'] * gpt_rt_audio_input_cost / 1000000,
        'cached_audio_input': tokens['input_audio_tokens_cached'] * gpt_rt_cached_audio_input_cost / 1000000,
        'audio_output': tokens['output_audio_tokens'] * gpt_rt_audio_output_cost / 1000000,
        'text_input': tokens['input_text_tokens'] * gpt_rt_text_input_cost / 1000000,
        'cached_text_input': tokens['input_text_tokens_cached'] * gpt_rt_cached_text_input_cost / 1000000,
        'text_output': tokens['output_text_tokens'] * gpt_rt_text_output_cost / 1000000
    }

    # Calculate GPT-4 Mini Realtime costs
    gpt_mini_rt_costs = {
        'audio_input': tokens['input_audio_tokens'] * gpt_mini_rt_audio_input_cost / 1000000,
        'cached_audio_input': tokens['input_audio_tokens_cached'] * gpt_mini_rt_cached_audio_input_cost / 1000000,
        'audio_output': tokens['output_audio_tokens'] * gpt_mini_rt_audio_output_cost / 1000000,
        'text_input': tokens['input_text_tokens'] * gpt_mini_rt_text_input_cost / 1000000,
        'cached_text_input': tokens['input_text_tokens_cached'] * gpt_mini_rt_cached_text_input_cost / 1000000,
        'text_output': tokens['output_text_tokens'] * gpt_mini_rt_text_output_cost / 1000000
    }

    return gpt_rt_costs, gpt_mini_rt_costs

def main():
    parser = argparse.ArgumentParser(description='Count tokens and calculate costs from JSON log file')
    parser.add_argument('file', help='Path to the JSON log file')
    args = parser.parse_args()

    # Extract token counts
    tokens = extract_token_counts(args.file)

    # Calculate costs
    gpt_rt_costs, gpt_mini_rt_costs = calculate_costs(tokens)

    # Print results (matching original shell script output format)
    print(f"{tokens['input_audio_tokens']} audio input tokens (${gpt_rt_costs['audio_input']:.3f}, ${gpt_mini_rt_costs['audio_input']:.3f})")
    print(f"{tokens['input_text_tokens']} text input tokens (${gpt_rt_costs['text_input']:.3f}, ${gpt_mini_rt_costs['text_input']:.3f})")
    print(f"{tokens['input_audio_tokens_cached']} cached audio input tokens (${gpt_rt_costs['cached_audio_input']:.3f}, ${gpt_mini_rt_costs['cached_audio_input']:.3f})")
    print(f"{tokens['input_text_tokens_cached']} cached text input tokens (${gpt_rt_costs['cached_text_input']:.3f}, ${gpt_mini_rt_costs['cached_text_input']:.3f})")
    print(f"{tokens['output_audio_tokens']} audio output tokens (${gpt_rt_costs['audio_output']:.3f}, ${gpt_mini_rt_costs['audio_output']:.3f})")
    print(f"{tokens['output_text_tokens']} text output tokens (${gpt_rt_costs['text_output']:.3f}, ${gpt_mini_rt_costs['text_output']:.3f})")
    print(f"Total cost (gpt-rt):      ${(gpt_rt_costs['audio_input'] + gpt_rt_costs['text_input'] + gpt_rt_costs['cached_audio_input'] + gpt_rt_costs['cached_text_input'] + gpt_rt_costs['audio_output'] + gpt_rt_costs['text_output']):.2f}")
    print(f"Total cost (gpt-mini-rt): ${(gpt_mini_rt_costs['audio_input'] + gpt_mini_rt_costs['text_input'] + gpt_mini_rt_costs['cached_audio_input'] + gpt_mini_rt_costs['cached_text_input'] + gpt_mini_rt_costs['audio_output'] + gpt_mini_rt_costs['text_output']):.2f}")

if __name__ == '__main__':
    main()
