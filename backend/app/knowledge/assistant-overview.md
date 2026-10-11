# The personal AI assistant

## What it is

Personal AI Assistant is a small conversational assistant. You chat with it in the browser, it streams its reply as it is written, and when a question needs a calculation, a lookup, the weather or a GIF it calls the matching tool and folds the result into its answer. A card in the chat shows each tool call while it runs.

## Languages

The assistant replies in the language your device is set to, whatever language the question is written in. The browser sends its language with each message, and the system prompt pins the reply to it, so a bare arithmetic expression gets an answer in Portuguese on a Portuguese device and in Spanish on a Spanish one. English is the default when no language is sent.

## Conversations

Past conversations are listed in the sidebar, most recently active first. A conversation is named after its first message unless you rename it, and it can be deleted. Replies that were stopped half-way are kept and marked as interrupted.

## Models

It can use Google's Gemini in the cloud or a model running on your own machine through LM Studio, chosen with one environment variable. With the local option no API key is needed and no data leaves the machine.
