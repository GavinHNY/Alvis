#pragma once
#include <map>
#include "Database.h"
#include "Llm.h"
#include "Windows.h"

class Alvis{
private:
	int c_maxChannel;

	Llm *m_llm;
	
	Database *m_db;

	int n_channel;
	std::vector<int> n_systemToken;
	std::vector<int> n_compressedHistory;
	std::vector<bool> channel;

public:
	struct textOutput {
		int clientId;
		std::string message;
	};


public:
	Alvis(Llm *llm, Database *db, int max_channel);
	
	void readSystemPrompt(std::string filepath);
	textOutput sendText(int pUserId, int clientId, std::string pText);

private:
	std::vector<std::string> systemPrompt;


};

