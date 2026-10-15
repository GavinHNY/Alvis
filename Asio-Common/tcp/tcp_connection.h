#pragma once

#include "common.h"
#include "tsQueue.h"
#include "message.h"

namespace net {
	template<typename T>
	class Connection : public std::enable_shared_from_this<Connection<T>> {
	public:
		enum class Owner {
			server,
			client
		};

		Connection(Owner parent, asio::io_context& asioContext, asio::ip::tcp::socket socket, tsQueue<owned_message<T>>& qIn)
			: m_asioContext(asioContext), m_socket(std::move(socket)), m_qMessagesIn(qIn) {
			m_nOwnerType = parent;
		}

		virtual ~Connection() {

		}

		uint32_t getID() const {
			return id;
		}


	public:
		void connectToClient(uint32_t uid = 0) {
			if (m_nOwnerType == Owner::server) {
				if (m_socket.is_open()) {
					id = uid;
					readHeader();
				}
			}
		}

		void connectToServer(const asio::ip::tcp::resolver::results_type& endpoints) {
			if (m_nOwnerType == Owner::client) {
				asio::async_connect(m_socket, endpoints, [this](std::error_code ec, asio::ip::tcp::endpoint endpoints) {
					if (!ec) {
						readHeader();
					} else {
						std::cout << "[CLIENT] Connect to Server Fail.\n";
					}
					});
			}
		}

		void disconnect() {
			if (isConnected())
				asio::post(m_asioContext, [this]() { m_socket.close(); });
		}

		bool isConnected() const {
			return m_socket.is_open();
		}

		bool isAuthorized() const {
			return authorized;
		}

		void setAuthorized(bool pAuthorized) {
			authorized = pAuthorized;
		}


	public:
		void send(const message<T>& msg) {
			asio::post(m_asioContext, [this, msg]() {
				bool bWritingMessage = !m_qMessagesOut.empty();
				m_qMessagesOut.push_back(msg);

				if (!bWritingMessage)
					writeHeader();
				});
		}


		//ASYNC
		void readHeader(bool bReadOnce = false) {
			//std::cout << "read header\n";
			asio::async_read(m_socket, asio::buffer(&m_msgTemporaryIn.header, sizeof(message_header<T>)),
				[this, bReadOnce](std::error_code ec, std::size_t length) {
					if (!ec) {
						if (m_msgTemporaryIn.header.size > 0) {
							m_msgTemporaryIn.body.resize(m_msgTemporaryIn.header.size);
							
							readBody(bReadOnce);
						} else {
							addToIncomingMessageQueue(bReadOnce);
						}

					} else {
						std::cout << "[" << id << "] Read Header Fail.\n";
						m_socket.close();
					}
				});
		}

	private:

		//ASYNC
		void readBody(bool bReadOnce) {
			asio::async_read(m_socket, asio::buffer(m_msgTemporaryIn.body.data(), m_msgTemporaryIn.body.size()),
				[this, bReadOnce](std::error_code ec, std::size_t length) {
					if (!ec) {
						addToIncomingMessageQueue(bReadOnce);
					} else {
						std::cout << "[" << id << "] Read Body Fail.\n";
						m_socket.close();
					}
				});
		}

		//ASYNC
		void writeHeader() {
			asio::async_write(m_socket, asio::buffer(&m_qMessagesOut.front().header, sizeof(message_header<T>)),
				[this](std::error_code ec, std::size_t length) {
					//std::cout << "wrote header: " << m_qMessagesOut.front() << std::endl;
					if (!ec) {
						//std::cout << "\nwrote " << m_qMessagesOut.front() << std::endl;
						if (m_qMessagesOut.front().body.size() > 0) {
							writeBody();
						} else {
							m_qMessagesOut.pop_front();

							if (!m_qMessagesOut.empty()) {
								writeHeader();
							}
						}
					} else {
						std::cout << "[" << id << "] Write Header Fail.\n";
						m_socket.close();
					}
				});
		}

		//ASYNC
		void writeBody() {
			//std::cout << "sending body" << m_qMessagesOut.front().body.data() << std::endl;
			asio::async_write(m_socket, asio::buffer(m_qMessagesOut.front().body.data(), m_qMessagesOut.front().body.size()),
				[this](std::error_code ec, std::size_t length) {
					if (!ec) {
						m_qMessagesOut.pop_front();
						if (!m_qMessagesOut.empty()) {
							writeHeader();
						}
					} else {
						std::cout << "[" << id << "] Write Body Fail.\n";
						m_socket.close();
					}
				});
		}

		void addToIncomingMessageQueue(bool bReadOnce) {
			//std::cout << "added to incoming message queue!\n";
			if (m_nOwnerType == Owner::server) {
				m_qMessagesIn.push_back({ this->shared_from_this(), m_msgTemporaryIn });
			} else {
				m_qMessagesIn.push_back({ nullptr, m_msgTemporaryIn });
			}
			//if (!bReadOnce)
				readHeader();
		}




	protected:
		asio::ip::tcp::socket m_socket;

		asio::io_context& m_asioContext;

		tsQueue<message<T>> m_qMessagesOut;

		tsQueue<owned_message<T>>& m_qMessagesIn;
		message<T> m_msgTemporaryIn;


		Owner m_nOwnerType = Owner::server;
		uint32_t id = 0;

		bool authorized = false;

		public:
			std::shared_ptr<asio::steady_timer> m_timeoutTimer;
	};
}
