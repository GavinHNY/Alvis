#pragma once
#include "common.h"
#include "tsQueue.h"
#include "message.h"
#include "tcp_connection.h"

namespace net {
	template <typename T>
	class TcpClientInterface {
	public:
		TcpClientInterface() : m_socket(m_context) {

		}

		virtual ~TcpClientInterface() {
			disconnect();
		}


	public:
		bool connect(const std::string& host, const uint16_t port) {
			try {
				asio::ip::tcp::resolver resolver(m_context);
				asio::ip::tcp::resolver::results_type endpoints = resolver.resolve(host, std::to_string(port));

				m_connection = std::make_unique<Connection<T>>(Connection<T>::Owner::client, m_context, asio::ip::tcp::socket(m_context), m_qMessagesIn);

				m_connection->connectToServer(endpoints);

				thrContext = std::thread([this]() { m_context.run(); });

			} catch (std::exception e) {
				std::cerr << "Client Exception: " << e.what() << std::endl;
				return false;
			}
			return true;
		}

		void disconnect() {
			if (isConnected()) {
				m_connection->disconnect();
			}

			m_context.stop();
			if (thrContext.joinable())
				thrContext.join();

			m_connection.release();
		}

		bool isConnected() {

			if (m_connection)
				return m_connection->isConnected();
			else
				return false;
		}
	public:
		void send(const message<T>& msg) {
			if (isConnected()) {

				m_connection->send(msg);
			}
		}

		tsQueue<owned_message<T>>& incoming() {
			return m_qMessagesIn;
		}
	protected:
		asio::io_context m_context;
		std::thread thrContext;
		asio::ip::tcp::socket m_socket;
		std::unique_ptr<Connection<T>> m_connection;

	private:
		tsQueue<owned_message<T>> m_qMessagesIn;
	};
}