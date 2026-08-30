#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "Database.h"
#include "Llm.h"



int main() {
	Database database("..\\models\\Alvis.db");
	std::vector<Llm> LLM;




	//Create main model

	//read systemPromptFile
	std::ifstream systemPromptFile("..\\models\\system_prompt.txt");
	std::stringstream systemPromptBuf;
	systemPromptBuf << systemPromptFile.rdbuf();
	std::string systemPrompt = systemPromptBuf.str();

	//create LLM object
	LLM.emplace_back("..\\models\\gemma-4-E4B-it.gguf", systemPrompt, &database, 1);
	database.pushChat(1, -1, systemPrompt);
	database.pushChat(1, 0, LLM.at(0).getOutput());
	std::cout << std::endl;
	//main chat loop
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
				database.deleteChat(1);
				LLM.at(0).deleteMemory();
				std::cout << "done!\n";
				continue;
			}
			if (prompt == "/free"){
				LLM.at(0).freeKVAndInjectSummary();
				continue;
			}
			if (prompt == "/sum"){
				std::cout << "summary: " << LLM.at(0).summarizeChat(database.readChat(1, 2));
				continue;
			}
			std::string formattedPrompt = LLM.at(0).applyTemplate(prompt, false, true);
			//std::cout << formattedPrompt << std::endl;
			if (LLM.at(0).prompt(formattedPrompt, true) == 1){
				database.pushChat(1, 1, prompt);
				database.pushChat(1, 0, LLM.at(0).getOutput());
				std::cout << std::endl;
			}
		}
	}
}
