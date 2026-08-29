#pragma once
#include "sqlite3.h"
#include <iostream>
#include <string>

class Database{
private:
	sqlite3 *db;
	int rc;
	char *errMsg = 0;

public:
	Database(std::string db_path);
	~Database();
	int pushChat(int channelId, int userId, std::string content);
	int deleteChat(int channelId);
	std::string readChat(int channelId, int from);
};

