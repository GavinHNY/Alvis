#include <iostream>
#include "net.h"

bool running = true;


enum class msgTypes : uint32_t {
	key,
	text,
	sound,
	ping,
};

class TcpClient : public net::TcpClientInterface<msgTypes> {
public:
	void authenticate(std::string key) {
		net::message<msgTypes> msg;
		msg.header.id = msgTypes::key;
		msg << key;
		send(msg);
	}

	void messageAll(std::string messages) {
		net::message<msgTypes> msg;
		msg.header.id = msgTypes::text;
		msg << messages;
		send(msg);
		
	}

	void ping() {
		net::message<msgTypes> msg;
		msg.header.id = msgTypes::ping;
		send(msg);
	}
	void message(std::string message) {
		net::message<msgTypes> msg;
		msg.header.id = msgTypes::text;
		msg << message;
		send(msg);
	}

};

void inputThread(TcpClient* tcpClient) {
	while (running) {
		std::string textInput;
		std::getline(std::cin, textInput);

		if (textInput[0] == '/') {
			if (textInput == "/ping") {
				tcpClient->ping();
			}
		} else {
			tcpClient->message(textInput);

		}
	}
}


int main() {
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);
	std::string key;
	std::cout << "password: \n";
	std::getline(std::cin, key);

	TcpClient tcpClient;
	tcpClient.connect("127.0.0.1", 41715);
	std::cout << "trying to authenticate\n";
	tcpClient.authenticate(key);

	std::thread input(inputThread, &tcpClient);

	while (running) {
		if (tcpClient.isConnected()) {
			if (!tcpClient.incoming().empty()) {
				auto msg = tcpClient.incoming().pop_front().msg;

				switch (msg.header.id) {
				case msgTypes::text:
				{
					std::string text;
					msg >> text;
					std::cout << text << std::endl;
				}
					break;
				case msgTypes::ping:
				{
					std::cout << "[SERVER] pong!\n";
				}
					break;
				}
			}
		}
	}
	
	input.join();
	input.detach();

	return 0;
}