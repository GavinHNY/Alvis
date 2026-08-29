#pragma once
#include "Database.h"
#include "common\chat.h"
#include "llama-cpp.h"
#include <string>
#include <iostream>
#include <vector>
#include <Windows.h>


class Llm
{	
private:
	int channelId;
	std::string sysPrompt;
	std::string modelPath;

	int n_ctx = 8192;
	int n_ctx_used;
	int n_sysToken;

	Database *database;

	llama_model *model;
	llama_sampler *smpl;
	llama_context *ctx;
	llama_memory_t mem;

	common_chat_templates_ptr tmpls;


	const struct llama_vocab *vocab;
	const struct ChatMessage{
		std::string role;
		std::string content;
	};

	std::string output;

	int injectSummary(std::string summarized_chat);

public:
	Llm(Llm &&) noexcept = default;
	Llm &operator=(Llm &&) noexcept = default;

	// Disable copy explicitly (optional but good practice)
	Llm(const Llm &) = delete;
	Llm &operator=(const Llm &) = delete;



	Llm(const std::string& model_path, const std::string& system_prompt, Database *database, int channelId, bool add_generation_prompt = true);
	~Llm();

	std::string applyTemplate(const std::string prompt, bool is_system = false, bool add_generation_prompt = false);
	int prompt(std::string &prompt, bool print);
	std::string getOutput();
	
	int freeKVAndInjectSummary();
	std::string summarizeChat(std::string chatHistory);

	int deleteMemory();

};

