#include "Llm.h"
void custom_log_callback(enum ggml_log_level level, const char *text, void *user_data){
	// Show only error messages
	if (level == GGML_LOG_LEVEL_ERROR){
		std::cerr << "[llama.cpp Error] " << text;
	}
}

Llm::Llm(std::string modelsFolder, std::string modelName){
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
	llama_log_set(custom_log_callback, nullptr);
	llama_backend_init();

	m_modelsFolder = modelsFolder;

	llama_model_params modelParams = llama_model_default_params();

	modelParams.n_gpu_layers = 99;
	modelParams.main_gpu = 0;

	model = llama_model_load_from_file((m_modelsFolder + modelName).c_str(), modelParams);

	if (model == NULL){
		std::cerr << "Error: unable to load model";
		return;
	}


	vocab = llama_model_get_vocab(model);

	tmpls = common_chat_templates_init(model, "");

	if (!tmpls){
		throw std::runtime_error("Failed to initialize common_chat_templates from model.");
	}

	//std::cout << "n_sysToken: " << n_sysToken << std::endl;
	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = n_ctx;
	ctx_params.n_batch = 4096;
	ctx_params.n_ubatch = 512;
	ctx_params.no_perf = false;
	ctx_params.n_seq_max = 10;
	ctx_params.kv_unified = true;
	ctx = llama_init_from_model(model, ctx_params);

	if (!ctx){
		std::cerr << "Error: Unable to create context";
		return;
	}
	

	auto sparam = llama_sampler_chain_default_params();
	sparam.no_perf = false;

	smpl = llama_sampler_chain_init(sparam);

	llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
	llama_sampler_chain_add(smpl, llama_sampler_init_temp(1.0f));
	llama_sampler_chain_add(smpl, llama_sampler_init_top_k(20.0f));
	llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
	llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
	mem = llama_get_memory(ctx);
}


bool Llm::loadModelBin() {
	size_t n_tokens_out = 0;
	if (llama_state_load_file(ctx, (m_modelsFolder + "bin\\Alvis.bin").c_str(), nullptr, 0, &n_tokens_out)) {
		n_ctxUsed = llama_memory_seq_pos_max(mem, channelId);
		std::fstream file;
		file.open(m_modelsFolder + "bin\\data.bin", std::ios::in | std::ios::binary);
		file.read(reinterpret_cast<char *>(&n_sysToken), sizeof(n_sysToken));
		file.read(reinterpret_cast<char *>(&n_sumSysToken), sizeof(n_sysToken));
		file.read(reinterpret_cast<char *>(&summarizedChat), sizeof(n_sysToken));

		std::cout << "ctx loaded, " << n_ctxUsed << "/" << n_ctx << std::endl;
		return true;
	} else{
		return false;
	}
}


std::string Llm::applyTemplate(const std::string prompt, bool is_system, bool add_generation_prompt){
	std::vector<common_chat_msg> rawMessages;
	rawMessages.clear();
	rawMessages.reserve(prompt.size());
	rawMessages.push_back({is_system ? "system" : "user", prompt});
	
	common_chat_templates_inputs inputs;
	inputs.messages = rawMessages;
	inputs.add_generation_prompt = add_generation_prompt;
	inputs.use_jinja = true;
	inputs.enable_thinking = true;


	common_chat_params result = common_chat_templates_apply(tmpls.get(), inputs);

	return result.prompt;
}



int Llm::prompt(int seq, std::string &prompt, bool print, bool generate_output){
	//std::cout << prompt;
	int previousNTokens = llama_memory_seq_pos_max(mem, seq);
	auto startTime = llama_time_us();
	std::vector<llama_token> tokens(4096);
	llama_token n_tokens;
	if ((n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true)) < 0){
		GGML_ABORT( "Error: unable to tokenize");
	}
	tokens.resize(n_tokens);
	n_ctxUsed = llama_memory_seq_pos_max(mem, seq) + 1;
	llama_batch batch = llama_batch_init(n_tokens, 0, 1);
	batch.n_tokens = n_tokens;
	for (int i = 0; i < n_tokens; i++){
		batch.token[i] = tokens[i];
		batch.pos[i] = n_ctxUsed + i;
		batch.n_seq_id[i] = 1;
		batch.seq_id[i][0] = seq;
		batch.logits[i] = (i == (n_tokens - 1)) ? 1 : 0;
	}

	int ret = llama_decode(ctx, batch);
	if (ret != 0){
		GGML_ABORT("failed to decode, ret = %d\n", ret);
	}

	llama_batch_free(batch);

	if (!generate_output){
		return n_tokens;
	}

	std::cout << seq <<"Alvis: ";

	llama_token nextTokenId;
	output[seq] = "";
	llama_batch gen_batch = llama_batch_init(1, 0, 1);

	int startThinking = 0;
	int endThinking;
	int current_ctx;

	bool thinking = false;
	bool outputAnswer = false;
	std::string allResponse = "";
	while (true){
		n_ctx = llama_n_ctx(ctx);
		current_ctx = llama_memory_seq_pos_max(mem, seq);
		n_ctxUsed = current_ctx + 1;

		if (n_ctxUsed + gen_batch.n_tokens > n_ctx/1.3){
			freeKVAndInjectSummary();
			current_ctx = llama_memory_seq_pos_max(mem, seq);
			std::cout << "CTX overload, freeing KV\n";
		}

		nextTokenId = llama_sampler_sample(smpl, ctx, -1);
		llama_sampler_accept(smpl, nextTokenId);


		if (llama_vocab_is_eog(vocab, nextTokenId) || nextTokenId == llama_vocab_eot(vocab) || nextTokenId == llama_vocab_eos(vocab)){
			break;
		}

		char buf[256];
		int n = llama_token_to_piece(vocab, nextTokenId, buf, sizeof(buf), 0, true);
		if (n < 0){
			GGML_ABORT("failed to convert token to piece\n");
		}
		std::string piece(buf, n);
		allResponse += piece;

		if (piece == "<|channel>"){
			thinking = true;
			startThinking = startThinking == 0? current_ctx: startThinking;
			outputAnswer = false;

		}

		if (outputAnswer){
			output[seq] = output[seq] + piece;
			if (print){
				std::cout << piece.c_str() << std::flush;
			}
		}

		if (piece == "<channel|>"){
			thinking = false;
			endThinking = current_ctx;
			outputAnswer = true;
		}

		//batch = llama_batch_get_one(&nextTokenId, 1);
		//batch = llama_batch_init(1, 0, 1);
		gen_batch.n_tokens = 1;
		gen_batch.token[0] = nextTokenId;
		gen_batch.pos[0] = n_ctxUsed;
		gen_batch.n_seq_id[0] = 1;
		gen_batch.seq_id[0][0] = seq;
		gen_batch.logits[0] = 1;

		int ret = llama_decode(ctx, gen_batch);
		if (ret != 0){
			GGML_ABORT("failed to decode, ret = %d\n", ret);
		}
	}
	
	llama_batch_free(gen_batch);
	current_ctx = llama_memory_seq_pos_max(mem, seq);
	//std::cout << "\n ctx before strip:" << llama_memory_seq_pos_max(mem, seq) + 1 << "/" << n_ctx << std::endl << std::endl;
	//strip thought from KV
	//std::cout << "startThinking: " << startThinking << " endThinking: " << endThinking << std::endl;


	if (endThinking == 0 && generate_output){
		if (print){
			std::cerr << "\n \nError: No output" << allResponse << std::endl;
		}
		output[seq] = allResponse;
	} else{
		llama_memory_seq_rm(mem, seq, startThinking, endThinking + 1);
		llama_memory_seq_add(mem, seq, endThinking + 1, -1, -(endThinking - startThinking + 1));
	}

	//std::cout << startThinking - endThinking - 1;
	auto endTime = llama_time_us();
	std::cout<< "\nctx:" << llama_memory_seq_pos_max(mem, seq) + 1 << "/" << n_ctx << "time: " << (endTime - startTime)/1000000.0f << std::endl;
	//std::cout << std::endl;

	return llama_memory_seq_pos_max(mem, seq) - previousNTokens;
}



std::string Llm::getOutput(int channel){
	std::string tempOutput = output[channel];
	output[channel] = "";
	return tempOutput;
}

int Llm::freeKVAndInjectSummary(){

	llama_memory_seq_rm(mem, channelId, n_sysToken, -1);

	//llama_memory_seq_rm(mem, channelId, n_sysToken, llama_memory_seq_pos_max(mem, channelId) - 100 + 1);
	//llama_memory_seq_add(mem, channelId, llama_memory_seq_pos_max(mem, channelId) - 100, -1, -(llama_memory_seq_pos_max(mem, channelId) - 100 - n_sysToken + 1));
	injectSummary(summarizeChat(database->readChat(channelId, summarizedChat + 1)));
	summarizedChat = database->getNRow("chatHistory", "WHERE channelId = " + std::to_string(channelId));
	//std::cout << summarizedChat << std::endl;
	n_ctxUsed = llama_memory_seq_pos_max(mem, channelId) + 1;
	std::cout << "\n Free ed KV KV now:" << n_ctxUsed << std::endl;

	return 0;
}

std::string Llm::summarizeChat(std::string chatHistory){
	std::string formattedChatHistory = applyTemplate(chatHistory, false, true);
	prompt(0, formattedChatHistory, true, true);
	llama_memory_seq_rm(mem, 0, n_sumSysToken, -1);
	return getOutput(0);
}



int Llm::injectSummary(std::string summarized_chat){
	llama_tokens summaryToken(4096);
	int n_summaryToken = llama_tokenize(vocab, summarized_chat.c_str(), summarized_chat.length(), summaryToken.data(), summaryToken.size(), true, true);
	summaryToken.resize(n_summaryToken);
	llama_batch batch = llama_batch_init(n_summaryToken, 0, 1);
	batch.n_tokens = n_summaryToken;
	//llama_memory_seq_add(mem, channelId, n_sysToken , -1, n_summaryToken);
	std::cout << llama_memory_seq_pos_max(mem, channelId) <<std::endl;
	for (int i = 0; i < n_summaryToken; i++){
		batch.token[i] = summaryToken[i];
		batch.pos[i] = n_sysToken + i;
		batch.n_seq_id[i] = 1;
		batch.seq_id[i][0] = channelId;
		batch.logits[i] = 0;
	}
	int ret = llama_decode(ctx, batch);
	if (ret != 0){
		GGML_ABORT("failed to decode, ret = %d\n", ret);
	}

	llama_batch_free(batch);
	n_sysToken += n_summaryToken;
	//std::cout << "nSystemToken: " << n_sysToken << std::endl;
	return n_summaryToken;
}

int Llm::deleteMemory(){
	llama_memory_t mem = llama_get_memory(ctx);
	llama_memory_clear(mem, true);
	return 0;
}


Llm::~Llm(){
	std::cout << "saving LLM\n";
	n_ctxUsed = llama_memory_seq_pos_max(mem, channelId);
	if (n_ctxUsed > 0){
		if (llama_state_save_file(ctx, (m_modelsFolder + "bin\\Alvis.bin").c_str(), nullptr, 0)) {
			std::cout << "saved in "<< m_modelsFolder + "bin\\Alvis.bin  " << n_ctxUsed << " ctx.\n";
			std::fstream file;
			file.open(m_modelsFolder + "bin\\data.bin", std::ios::out | std::ios::binary);
			file.write(reinterpret_cast<char*>(&n_sysToken), sizeof(n_sysToken));
			file.write(reinterpret_cast<char*>(&n_sumSysToken), sizeof(n_sysToken));
			file.write(reinterpret_cast<char*>(&summarizedChat), sizeof(n_sysToken));


		} else{
			std::cerr << "failed to save ctx";
		}
	} else{
		std::filesystem::remove((m_modelsFolder + "bin\\Alvis.bin").c_str());
	}
	llama_sampler_free(smpl);
	llama_model_free(model);
	llama_free(ctx);
	llama_backend_free();
}