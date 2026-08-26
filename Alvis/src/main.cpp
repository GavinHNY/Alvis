#include <iostream>
#include <fstream>
#include "Llm.h"
#include "Database.h"



int main() {
	Database database("..\\models\\Alvis.db");




	std::ifstream systemPromptFile("..\\models\\system_prompt.txt");
	std::stringstream systemPromptBuf;
	systemPromptBuf << systemPromptFile.rdbuf();
	std::string systemPrompt = systemPromptBuf.str();

	Llm* LLM = new Llm("..\\models\\gemma-4-E4B-it.gguf", systemPrompt, database);

	bool running = true;

	std::string prompt;
	while (running) {
		std::cin.clear();
		std::getline(std::cin, prompt);
		if (prompt != ""){
			if (prompt == "/bye" || prompt == "/exit"){
				break;
			}
			if (prompt == "/clearChat" || prompt == "/deleteChat"){
				database.deleteChat();
				std::cout << "done!\n";
				continue;
			}
			std::string formattedPrompt = LLM->applyTemplate(prompt, false, true);
			//std::cout << formattedPrompt << std::endl;
			if (LLM->prompt(formattedPrompt, true) == 1){
				database.pushChat(0, 1, prompt);
				database.pushChat(0, 0, LLM->getOutput());
			}
		}
	}


	delete LLM;
}
