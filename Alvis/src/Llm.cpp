#include "Llm.h"

Llm::Llm(const std::string &model_path, const std::string &system_prompt, const std::string &sumSystem_prompt, Database *database, int channel_id, bool add_generation_prompt){
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
	llama_log_set([](ggml_log_level level, const char *text, void *user_data){}, nullptr);
	llama_backend_init();

	channelId = channel_id;
	this->database = database;
	sysPrompt = system_prompt;
	sumSysPrompt = sumSystem_prompt;
	modelPath = model_path;

	llama_model_params modelParams = llama_model_default_params();

	modelParams.n_gpu_layers = 99;
	modelParams.main_gpu = 0;

	model = llama_model_load_from_file(model_path.c_str(), modelParams);

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
	ctx_params.n_seq_max = 4;
	ctx_params.kv_unified = true;
	ctx = llama_init_from_model(model, ctx_params);

	if (!ctx){
		std::cerr << "Error: Unable to create context";
		return;
	}
	
	mem = llama_get_memory(ctx);

	auto sparam = llama_sampler_chain_default_params();
	sparam.no_perf = false;

	smpl = llama_sampler_chain_init(sparam);

	llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
	llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
	llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
	

	//std::cout << ready_sys_prompt << std::endl;
	summarizedChat = 1;

	std::string ready_sys_prompt = applyTemplate(system_prompt, true, add_generation_prompt);
	n_sysToken = -llama_tokenize(vocab, ready_sys_prompt.c_str(), ready_sys_prompt.length(), nullptr, 0, true, true);
	prompt(channelId, ready_sys_prompt);

	std::string ready_sumSys_prompt = applyTemplate(sumSystem_prompt, true, false);
	n_sumSysToken = -llama_tokenize(vocab, ready_sumSys_prompt.c_str(), ready_sumSys_prompt.length(), nullptr, 0, true, true);
	prompt(0, ready_sumSys_prompt, false, false);
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
	//std::cout << formatedPrompt;
	std::vector<llama_token> tokens(4096);
	llama_token n_tokens;
	if ((n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true)) < 0){
		GGML_ABORT( "Error: unable to tokenize");
	}
	tokens.resize(n_tokens);
	n_ctx_used = llama_memory_seq_pos_max(mem, seq) + 1;
	llama_batch batch = llama_batch_init(n_tokens, 0, 1);
	batch.n_tokens = n_tokens;
	for (int i = 0; i < n_tokens; i++){
		batch.token[i] = tokens[i];
		batch.pos[i] = n_ctx_used + i;
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
		return 1;
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
		n_ctx_used = current_ctx + 1;
		if (n_ctx_used + gen_batch.n_tokens > n_ctx){
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

		if (piece == "<channel|>" && thinking){
			thinking = false;
			endThinking = current_ctx;
			outputAnswer = true;
		}

		//batch = llama_batch_get_one(&nextTokenId, 1);
		//batch = llama_batch_init(1, 0, 1);
		gen_batch.n_tokens = 1;
		gen_batch.token[0] = nextTokenId;
		gen_batch.pos[0] = n_ctx_used;
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

	std::cout<< "\nctx:" << llama_memory_seq_pos_max(mem, seq) + 1 << "/" << n_ctx << std::endl;
	//std::cout << std::endl;

	return 1;
}



std::string Llm::getOutput(int channel){
	return output[channel];
}

int Llm::freeKVAndInjectSummary(){
	
	llama_memory_seq_rm(mem, channelId, n_sysToken, -1);

	//llama_memory_seq_rm(mem, channelId, n_sysToken, llama_memory_seq_pos_max(mem, channelId) - 100 + 1);
	//llama_memory_seq_add(mem, channelId, llama_memory_seq_pos_max(mem, channelId) - 100, -1, -(llama_memory_seq_pos_max(mem, channelId) - 100 - n_sysToken + 1));
	injectSummary(summarizeChat(database->readChat(channelId, summarizedChat + 1)));
	summarizedChat = database->getNRow("chatHistory", "WHERE channelId = " + std::to_string(channelId));
	//std::cout << summarizedChat << std::endl;
	n_ctx_used = llama_memory_seq_pos_max(mem, channelId) + 1;
	std::cout << "\n Free ed KV KV now:" << n_ctx_used << std::endl;

	return 0;
}

std::string Llm::summarizeChat(std::string chatHistory){
	std::string formattedChatHistory = applyTemplate(chatHistory, false, true);
	prompt(0, formattedChatHistory, true, true);
	llama_memory_seq_rm(mem, 0, n_sumSysToken, -1);
	return getOutput(0);
}

//std::string Llm::summarizeChat(std::string chat_history){
//	std::string prompt = sumSysPrompt + chat_history;
//
//	std::string formmated = applyTemplate(prompt, true, true);
//	//std::cout << formmated << std::endl;
//	int n_p = -llama_tokenize(vocab, formmated.c_str(), formmated.length(), nullptr, 0, true, true);
//	std::vector<llama_token> p_token(n_p + 16);
//	llama_tokenize(vocab, formmated.c_str(), formmated.length(), p_token.data(), p_token.size(), true, true);
//	p_token.resize(n_p);
//	//std::cout << "n_p: " << n_p << std::endl;
//	llama_batch batch = llama_batch_init(n_p, 0, 1);
//	batch.n_tokens = n_p;
//	for (int i = 0; i < n_p; i++){
//		batch.token[i] = p_token[i];
//		batch.pos[i] = i;
//		batch.n_seq_id[i] = 1;
//		batch.seq_id[i][0] = 0;
//		batch.logits[i] = (i == (n_p - 1)) ? 1 : 0;
//	}
//	int ret = llama_decode(ctx, batch);
//	if (ret != 0){
//		GGML_ABORT("failed to decode, ret = %d\n", ret);
//	}
//	llama_batch_free(batch);
//
//	int seq2_pos = n_p;
//
//	std::string summary = "";
//	bool output = false;
//	while (true){
//		llama_token newTokenId = llama_sampler_sample(smpl, ctx, -1);
//		llama_sampler_accept(smpl, newTokenId);
//		//std::cout << "test 1\n";
//
//		if (llama_vocab_is_eog(vocab, newTokenId)){
//			break;
//		}
//
//		//std::cout << "test 2\n";
//		char buf[256];
//		int n = llama_token_to_piece(vocab, newTokenId, buf, sizeof(buf), 0, true);
//		if (n < 0){
//			GGML_ABORT("failed to convert token to piece\n");
//		}
//		std::string piece(buf, n);
//		if (output){
//			std::cout << piece << std::flush;
//			summary += piece;
//		}
//		//std::cout << "piece: " << piece << std::endl;
//		
//		if (piece == "<channel|>") output = true;
//	
//		llama_batch gen_batch = llama_batch_init(1, 0, 1);
//		gen_batch.n_tokens = 1;
//		gen_batch.token[0] = newTokenId;
//		gen_batch.n_seq_id[0] = 1;
//		gen_batch.seq_id[0][0] = 0;
//		gen_batch.pos[0] = seq2_pos++;
//		gen_batch.logits[0] = 1;
//
//		llama_decode(ctx, gen_batch);
//		llama_batch_free(gen_batch);
//	}
//
//
//	llama_memory_seq_rm(mem, 0, -1, -1);
//	//std::cout << summary <<std::endl;
//	return summary;
//}

int Llm::injectSummary(std::string summarized_chat){
	std::vector<llama_token> summaryToken(4096);
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
	llama_sampler_free(smpl);
	llama_model_free(model);
	llama_free(ctx);
	llama_backend_free();
}