#include "Database.h"
#include <format>

Database::Database(std::string db_path){
	
	rc = sqlite3_open(db_path.c_str(), &db);
	if (rc != SQLITE_OK){
		std::cerr << "Error: Cannot open database: "<< sqlite3_errmsg(db);
		abort;
	}


	std::string sql;
	
	//create chatHistory table
	 sql = R"(
CREATE TABLE IF NOT EXISTS chatHistory(
id INTEGER PRIMARY KEY AUTOINCREMENT,
channelId INTEGER NOT NULL,
userId INTEGER NOT NULL,
datetime TEXT NOT NULL,
content TEXT NOT NULL
);)";

	 rc = sqlite3_exec(db, sql.c_str(), 0, 0, &errMsg);
	 if (rc != SQLITE_OK){
		 std::cerr << "Error: Failed to create chatHistory table:" << errMsg << std::endl;
		 sqlite3_free(errMsg);
		 abort();
	 }
}

Database::~Database(){
	sqlite3_close(db);
}

int Database::pushChat(int channelId, int userId, std::string content){
	std::string sql;
	sql = "insert into chatHistory values(NULL, ?, ?, datetime(), ?);";
	sqlite3_stmt *stmt;
	sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, NULL);
	sqlite3_bind_int(stmt, 1, channelId);
	sqlite3_bind_int(stmt, 2, userId);
	sqlite3_bind_text(stmt, 3, content.c_str(), -1, SQLITE_STATIC);
	//std::cout << content.c_str() << std::endl;
	if (sqlite3_step(stmt) != SQLITE_DONE){
		std::cerr << "Error: Failed to insert into chatHistory table" << sqlite3_errmsg << std::endl;
		abort();
	}
	sqlite3_finalize(stmt);


	return 0;
}

int Database::deleteChat(){
	sqlite3_exec(db, "DROP TABLE chatHistory;", 0, 0, NULL);
	return 0;
}



