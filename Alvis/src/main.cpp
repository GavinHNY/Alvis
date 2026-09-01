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

	std::ifstream sumSystemPromptFile("..\\models\\sum_system_prompt.txt");
	std::stringstream sumSystemPromptBuf;
	sumSystemPromptBuf << sumSystemPromptFile.rdbuf();
	std::string sumSystemPrompt = sumSystemPromptBuf.str();

	//create LLM object
	int mainChannel = 1;

	LLM.emplace_back("..\\models\\gemma-4-E4B-it.gguf", systemPrompt, sumSystemPrompt, &database, mainChannel);
	database.pushChat(mainChannel, -1, systemPrompt);
	database.pushChat(mainChannel, 0, LLM.at(0).getOutput(mainChannel));
	std::cout << std::endl;
	//main chat loop
	bool running = true;

	std::string prompt;
	while (running) {
		std::cin.clear();
		std::cout << "User: ";
		std::getline(std::cin, prompt);
		if (prompt != ""){
			if (prompt == "/bye" || prompt == "/exit"){
				break;
			}
			if (prompt == "/clearChat" || prompt == "/deleteChat"){
				database.deleteChat(mainChannel);
				LLM.at(0).deleteMemory();
				std::cout << "done\n";
				continue;
			}
			if (prompt == "/free"){
				LLM.at(0).freeKVAndInjectSummary();
				continue;
			}
			if (prompt == "/sum"){
				std::cout << "summary: " << LLM.at(0).summarizeChat(database.readChat(mainChannel, 2));
				continue;
			}
			if (prompt == "/count"){
				std::cout << database.getNRow("chatHistory", "WHERE channelId = " + std::to_string(mainChannel)) << std::endl;
				continue;
			}

			if (prompt[0] == '/'){
				std::cout << "Wrong command. \n";
				continue;
			}

			std::string formattedPrompt = LLM.at(0).applyTemplate(prompt, false, true);
			//std::cout << formattedPrompt << std::endl;
			if (LLM.at(0).prompt(mainChannel, formattedPrompt, true, true) == 1){
				database.pushChat(mainChannel, 1, prompt);
				database.pushChat(mainChannel, 0, LLM.at(0).getOutput(mainChannel));
				std::cout << std::endl;
			}
		}
	}
}
