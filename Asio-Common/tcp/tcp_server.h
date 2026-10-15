#pragma once

#include "common.h"
#include "tsQueue.h"
#include "message.h"
#include "tcp_connection.h"

namespace net {
	template<typename T>
	class TcpServerInterface {
	public:
		TcpServerInterface(uint16_t port) : m_asioAcceptor(m_asioContext, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port)) {

		}

		virtual ~TcpServerInterface() {
			stop();
		}

		std::shared_ptr<Connection<T>> getClient(uint32_t id) {
			return m_umapConnections.at(id);
		}

		bool start() {
			try {
				waitForClientConnection();

				m_threadContext = std::thread([this]() { m_asioContext.run(); });

			} catch (std::exception& e) {
				std::cerr << "[SERVER} Exception: " << e.what() << "\n";
				return false;
			}

			std::cout << "[SERVER] Started! \n";
			return true;
		}

		bool stop() {
			m_asioContext.stop();

			if (m_threadContext.joinable()) m_threadContext.join();

			std::cout << "[SERVER] Stopped!\n";
			return true;
		}

		//ASYNC
		void waitForClientConnection() {
			m_asioAcceptor.async_accept([this](std::error_code ec, asio::ip::tcp::socket socket) {
				if (!ec) {
					std::cout << "[SERVER] New Connection: " << socket.remote_endpoint() << std::endl;

					std::shared_ptr<Connection<T>> newconn = std::make_shared<Connection<T>>(Connection<T>::Owner::server, m_asioContext, std::move(socket), m_qMessagesIn);

					if (true) {
						m_umapConnections[nIDCounter] = std::move(newconn);
						m_umapConnections.at(nIDCounter)->connectToClient(nIDCounter);
						onClientConnect(m_umapConnections.at(nIDCounter++));
						
					} else {
						std::cout << "[-----] connection Denied\n";
					}

				} else {
					std::cout << "[SERVER] New Connection Error: " << ec.message() << std::endl;
				}
				waitForClientConnection();
				});
		}

		void readValidation(std::shared_ptr<Connection<T>> client) {
			client->readHeader(true);
		}

		void messageClient(std::shared_ptr<Connection<T>> client, const message<T>& msg) {
			if (client && client->isConnected()) {
				client->send(msg);
			} else {
				onClientDisconnect(client);
				client.reset();
				m_umapConnections.erase(client->getID());
			}
		}

		void messageAllClient(const message<T>& msg, std::shared_ptr<Connection<T>> pIgnoreClient = nullptr) {
			bool bInvalidClientExists = false;
			for (auto& [id, client] : m_umapConnections) {
				if (client && client->isConnected()) {
					if (client != pIgnoreClient)
						client->send(msg);
				} else {
					onClientDisconnect(client);
					client.reset();
					bInvalidClientExists = true;
				}
			}

			if (bInvalidClientExists) {
				std::erase_if(m_umapConnections, [](const auto& pair) {
					return pair.second == nullptr;
					});
			}

		}

		void update(size_t nMaxMessages = -1, bool bWait = false) {

			if (bWait) m_qMessagesIn.wait();

			/*if (m_qMessagesIn.count() > 0)
				std::cout << "New Update: number of message: " << m_qMessagesIn.count() << std::endl;*/

			size_t nMessageCount = 0;
			
			while (nMessageCount < nMaxMessages && !m_qMessagesIn.empty()) {
				//std::cout << "update!\n";
				auto msg = m_qMessagesIn.pop_front();

				onMessage(msg.remote, msg.msg);

				nMessageCount++;
			}
		}
	protected:
		virtual void onClientConnect(std::shared_ptr<Connection<T>> client) {}

		virtual void onClientDisconnect(std::shared_ptr<Connection<T>> client) {} //Verify

		virtual void onMessage(std::shared_ptr<Connection<T>> client, message<T>& msg) {}


	protected:
		tsQueue<owned_message<T>> m_qMessagesIn;

		std::unordered_map<uint32_t, std::shared_ptr<Connection<T>>> m_umapConnections;

		asio::io_context m_asioContext;
		std::thread m_threadContext;

		asio::ip::tcp::acceptor m_asioAcceptor;
		uint32_t nIDCounter = 10000;


	};
}