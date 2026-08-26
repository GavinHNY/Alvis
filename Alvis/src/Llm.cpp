#include "Llm.h"

Llm::Llm(const std::string &model_path, const std::string &system_prompt, Database &database){
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
	llama_log_set([](ggml_log_level level, const char *text, void *user_data){}, nullptr);
	llama_backend_init();


	llama_model_params modelParams = llama_model_default_params();

	modelParams.n_gpu_layers = 99;

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

	std::string ready_sys_prompt = applyTemplate(system_prompt, true, true);

	//std::cout << ready_sys_prompt << std::endl;

	n_sysToken = -llama_tokenize(vocab, ready_sys_prompt.c_str(), ready_sys_prompt.length(), nullptr, 0, true, true);
	//std::cout << "n_sysToken: " << n_sysToken << std::endl;
	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 16384;
	ctx_params.n_batch = 16384;
	ctx_params.no_perf = false;
	ctx = llama_init_from_model(model, ctx_params);

	if (!ctx){
		std::cerr << "Error: Unable to create context";
		return;
	}

	auto sparam = llama_sampler_chain_default_params();
	sparam.no_perf = false;

	smpl = llama_sampler_chain_init(sparam);

	llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
	llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
	llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

	if (prompt(ready_sys_prompt, true)){
		database.pushChat(0, 0, getOutput());
	}
}




std::string Llm::applyTemplate(const std::string prompt, bool is_system, bool add_generation_prompt){
	std::vector<common_chat_msg> rawMessages;
	rawMessages.reserve(prompt.size() + chatHistory.size());
	chatHistory.push_back({is_system ? "system" : "user", prompt});

	for (const auto &msg : chatHistory){
		rawMessages.push_back({msg.role.c_str(), msg.content.c_str()});
	}

	common_chat_templates_inputs inputs;
	inputs.messages = rawMessages;
	inputs.add_generation_prompt = add_generation_prompt;
	inputs.use_jinja = true;
	inputs.enable_thinking = true;


	common_chat_params result = common_chat_templates_apply(tmpls.get(), inputs);

	return result.prompt;
}



int Llm::prompt(std::string &formatedPrompt, bool print){
	std::vector<llama_token> sys_tokens(formatedPrompt.length() + 16);

	llama_token n_tokens = -llama_tokenize(vocab, formatedPrompt.c_str(), formatedPrompt.length(), NULL, 0, true, true);
	std::vector<llama_token> tokens(n_tokens + 16);
	if (llama_tokenize(vocab, formatedPrompt.c_str(), formatedPrompt.length(), tokens.data(), tokens.size(), true, true) < 0){
		GGML_ABORT( "Error: unable to tokenize");
	}
	tokens.resize(n_tokens);
	llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

	
	llama_token nextTokenId;
	output = "";

	bool outputAnswer = false;
	while (true){
		int n_ctx = llama_n_ctx(ctx);
		int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
		if (n_ctx_used + batch.n_tokens > n_ctx){
			printf("\033[0m\n");
			fprintf(stderr, "context size exceeded\n");
			exit(0);
		}
		int ret = llama_decode(ctx, batch);
		if (ret != 0){
			GGML_ABORT("failed to decode, ret = %d\n", ret);
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
		if (outputAnswer){
			output = output + piece;
			if (print){
			std::cout << piece.c_str()<< std::flush;
			}
		}
		if (piece == "<channel|>") outputAnswer = true;

		batch = llama_batch_get_one(&nextTokenId, 1);
	}
	std::cout << std::endl;

	return 1;
}



std::string Llm::getOutput(){
	return output;
}



Llm::~Llm(){
	llama_sampler_free(smpl);
	llama_model_free(model);
	llama_free(ctx);
	llama_backend_free();
}




