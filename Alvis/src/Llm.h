#pragma once
#include "Database.h"
#include "common\chat.h"
#include "llama-cpp.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <filesystem>
#include <Windows.h>


class Llm
{	
private:
	int channelId;
	std::string sysPrompt;
	std::string sumSysPrompt;
	std::string modelPath;

	int n_ctx = 32768;
	int n_ctxUsed;
	int n_sysToken;
	int n_sumSysToken;
	int summarizedChat;

	Database *database;

	llama_model *model;
	llama_sampler *smpl;
	llama_context *ctx;
	llama_memory_t mem;
	llama_tokens historyTokens;

	common_chat_templates_ptr tmpls;


	const struct llama_vocab *vocab;
	const struct ChatMessage{
		std::string role;
		std::string content;
	};

	std::map<int, std::string> output;

	int injectSummary(std::string summarized_chat);

	std::string ctxBin = "..\\models\\bin\\Alvis.bin";

public:
	Llm(Llm &&) noexcept = default;
	Llm &operator=(Llm &&) noexcept = default;

	// Disable copy explicitly (optional but good practice)
	Llm(const Llm &) = delete;
	Llm &operator=(const Llm &) = delete;



	Llm(const std::string& model_path, const std::string& system_prompt, const std::string &sumSystem_prompt, Database *database, int channel_id, bool add_generation_prompt = true);
	~Llm();

	std::string applyTemplate(const std::string prompt, bool is_system = false, bool add_generation_prompt = true);
	int prompt(int seq, std::string &prompt, bool print = true, bool generate_output = true);
	std::string getOutput(int channel);
	
	int freeKVAndInjectSummary();
	std::string summarizeChat(std::string chatHistory);

	int deleteMemory();

};

