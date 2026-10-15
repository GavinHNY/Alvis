#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "Database.h"
#include "net.h"
#include "Alvis.h"

bool running = true;
std::string modelsFolder = "D:\\dev\\Alvis\\models\\";

enum class msgTypes : uint32_t {
	key,
	text,
	sound,
	ping,
};

std::string key;




class TcpServer : public net::TcpServerInterface<msgTypes> {
public:
	TcpServer(uint16_t nPort, Alvis* alvis) : net::TcpServerInterface<msgTypes>(nPort), m_alvis(alvis){}

	void messageAll(std::string text) {
		net::message<msgTypes> msg;
		msg.header.id = msgTypes::text;
		msg << text;
		messageAllClient(msg);
	}

	void notifyQuit() {
		m_qMessagesIn.notifyQuit();
	}

protected:
	void onClientConnect(std::shared_ptr<net::Connection<msgTypes>> client) override{
		client->m_timeoutTimer = std::make_shared<asio::steady_timer>(m_asioContext, asio::chrono::seconds(30));
		client->m_timeoutTimer->async_wait([this, client](const std::error_code& ec) {
			if (ec == asio::error::operation_aborted && client->isAuthorized()) {
				std::cout << "[" << client->getID() << "] Connection Approved\n";
				return;
			} else {
				std::cout << "[" << client->getID() << "] Connection Not Approved\n";
				client->disconnect();
			}
		});
	}

	void onClientDisconnect(std::shared_ptr<net::Connection<msgTypes>> client) override {

	}

	void onMessage(std::shared_ptr<net::Connection<msgTypes>> client, net::message<msgTypes>& msg) override{
		//std::cout << "new messages\n";
		switch (msg.header.id) {
		case msgTypes::key:
		{
			//std::cout << "[" << client->getID() << "]: " << "recieved key\n";
			std::string clientKey(msg.body.begin(), msg.body.end());
			//msg >> clientKey;
			//std::cout << "[" << client->getID() << "]: " << "recieved key:" << clientKey << std::endl;
			if (std::strcmp(clientKey.c_str(), key.c_str())==0) {
				client->setAuthorized(true);
				client->m_timeoutTimer->cancel();
			} else {
				client->setAuthorized(false);
				//std::cout << "[" << client->getID() << "] client key: " << clientKey << " server key: "<< key <<"  compare " << (std::strcmp(clientKey.c_str(), key.c_str()) == 0)<< std::endl;
				client->disconnect();
				client->m_timeoutTimer->cancel();
			}
		}
		break;

		case msgTypes::ping:
		{
			std::cout << "[" << client->getID() << "]: ping!\n";
			net::message<msgTypes> rMsg;
			rMsg.header.id = msgTypes::ping;
			messageClient(client, rMsg);
		}
		break;

		case msgTypes::text:
		{
			std::string text;
			msg >> text;
			std::cout << "[" << client->getID() << "]: " << text << std::endl;
			net::message<msgTypes> rMsg;
			rMsg.header.id = msgTypes::text;
			std::string reply = m_alvis->sendText(1, client->getID(), text).message;
			rMsg << reply;
			std::cout << "[Alvis]: " << reply << std::endl;
			messageClient(client, rMsg);
		}
		break;
		}
	}




private:
	Alvis* m_alvis;
};

void serverConsoleInput(Database& database, Llm& LLM, int mainChannel, TcpServer& tcpServer) {
	std::string prompt;
	while (running) {
		std::cin.clear();
		std::getline(std::cin, prompt);
		if (prompt != "") {
			if (prompt == "/bye" || prompt == "/exit") {
				running = false;
				tcpServer.notifyQuit();
				return;
			}
			if (prompt == "/clearChat" || prompt == "/deleteChat") {
				database.deleteChat(mainChannel);
				LLM.deleteMemory();
				std::cout << "done\n";
				continue;
			}
			if (prompt == "/free") {
				LLM.freeKVAndInjectSummary();
				continue;
			}
			if (prompt == "/sum") {
				std::cout << "summary: " << LLM.summarizeChat(database.readChat(mainChannel, 2));
				continue;
			}
			if (prompt == "/count") {
				std::cout << database.getNRow("chatHistory", "WHERE channelId = " + std::to_string(mainChannel)) << std::endl;
				continue;
			}

			if (prompt[0] == '/') {
				std::cout << "unkown command. \n";
				continue;
			} else {
				tcpServer.messageAll(prompt);
			}
		}
	}
	return;
}




int main() {

	Database database(modelsFolder + "Alvis.db");
	//Create main model
	// 
		//create LLM object
	int mainChannel = 1;
	Llm LLM(modelsFolder, "gemma-4-E4B-it.gguf");
	Alvis alvis(&LLM, &database, 10);

	if (!alvis.loadModelBin(modelsFolder)) {
		alvis.readSystemPrompt(modelsFolder + "system_prompt.txt");
		alvis.readSystemPrompt(modelsFolder + "sum_system_prompt.txt");
	}


	std::cout << "please set a pasword: \n";
	std::getline(std::cin, key);
	std::cout << "key setted to: " << key << std::endl;
	TcpServer tcpServer((uint16_t)41715, &alvis);
	tcpServer.start();
	std::thread serverConsoleInputThread(serverConsoleInput, std::ref(database), std::ref(LLM), mainChannel, std::ref(tcpServer));

	std::cout << std::endl;
	//main chat loop

	while (running) {
		tcpServer.update(-1, true);
	}
	if (serverConsoleInputThread.joinable()){
		std::cout << "trying to detach\n";
		serverConsoleInputThread.detach();
	}
	std::cout << "stopping\n";

			//std::string formattedPrompt = LLM.applyTemplate(prompt, false, true);
			////std::cout << formattedPrompt << std::endl;
			//if (LLM.prompt(mainChannel, formattedPrompt, true, true) == 1){
			//	database.pushChat(mainChannel, 1, prompt);
			//	database.pushChat(mainChannel, 0, LLM.getOutput(mainChannel));
			//	std::cout << std::endl;
			//}
}