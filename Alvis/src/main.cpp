#include <iostream>
#include "Llm.h"



int main() {
	Llm* LLM = new Llm("..\\models\\gemma-4-E4B-it.gguf", "Your name is Alvis, An AI assistant made by Dragon. Your goal is to assist and chat with the user. Use short natual human spoken language so no emoji, and role play an actual human assistant but more as a friend. Use the LOW thinking system, think at a lower depth. Reply with a self introduction and ask for anything to help if you understand this.");

	bool running = true;

	std::string prompt;
	while (running) {
		std::cin.clear();
		std::getline(std::cin, prompt);
		if (prompt != ""){
			if (prompt == "/bye" || prompt == "/exit"){
				break;
			}
			std::string formattedPrompt = LLM->applyTemplate(prompt);
			LLM->prompt(formattedPrompt, true);
		}
	}


	delete LLM;
}
