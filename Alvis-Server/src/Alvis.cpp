#include "Alvis.h"

Alvis::Alvis(Llm *llm, Database *db, int max_channel){
	m_llm = llm;
	m_db = db;
	channel.resize(max_channel);
	n_systemToken.resize(max_channel);
	n_compressedHistory.resize(max_channel);
}

void Alvis::readSystemPrompt(std::string filepath){
	std::ifstream systemPromptFile(filepath);
	std::stringstream systemPromptBuf;
	systemPromptBuf << systemPromptFile.rdbuf();
	systemPrompt.push_back(systemPromptBuf.str());
	m_llm->prompt(1, systemPrompt[0], false, false);
	//m_llm->prompt(0, systemPrompt[1], false, false);
}

bool Alvis::loadModelBin(std::string modelsFolder){
	return m_llm->loadModelBin();
}



Alvis::textOutput Alvis::sendText(int pUserId, int clientId, std::string pText){
	std::string formattedPrompt = m_llm->applyTemplate(pText);
	m_llm->prompt(1, formattedPrompt, false);
	textOutput output(clientId, m_llm->getOutput(1));
	return output;
}
