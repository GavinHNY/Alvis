#pragma once
//#include "llama.h"
#include "llama-cpp.h"
#include "common\chat.h"
#include <string>
#include <iostream>
#include <vector>
#include <Windows.h>


class Llm
{	
private:
	llama_model *model;
	llama_sampler *smpl;
	llama_context *ctx;

	common_chat_templates_ptr tmpls;


	const struct llama_vocab *vocab;
	const struct ChatMessage{
		std::string role;
		std::string content;
	};
	std::vector<ChatMessage> chatHistory;

	int n_sysToken;
	std::string output;

	

public:
	Llm(const std::string& model_path, const std::string& system_prompt);
	~Llm();

	std::string applyTemplate(const std::string prompt, bool is_system = false, bool add_generation_prompt = false);
	int prompt(std::string &prompt, bool print);
	std::string getOutput();




};

