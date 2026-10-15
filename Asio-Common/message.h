#pragma once
#include "common.h"

namespace net {
	template <typename T>
	struct message_header {
		T id{};
		uint32_t size = 0;
	};

	template <typename T>
	struct message {
		message_header<T> header{};
		std::vector<uint8_t> body;

		size_t size() const {
			return body.size();
		}

		friend std::ostream& operator << (std::ostream& os, const message<T>& msg) {
			os << "ID: " << int(msg.header.id) << "  Size: " << msg.body.size();
			return os;
		}

		template <typename DataType>
		friend message<T>& operator << (message<T>& msg, const DataType& data) {
			static_assert(std::is_standard_layout<DataType>::value, "Data is too complex to be pushed into vector");

			size_t i = msg.body.size();
			msg.body.resize(msg.body.size() + sizeof(data));
			std::memcpy(msg.body.data() + i, &data, sizeof(data));

			msg.header.size = msg.size();
			return msg;
		}
		friend message<T>& operator << (message<T>& msg, const std::string& data) {
			size_t i = msg.body.size();
			msg.body.resize(msg.body.size() + data.size() * sizeof(char));
			std::memcpy(msg.body.data() + i, data.data(), data.size() * sizeof(char));
			//std::cout << "msg.size() = " << msg.size() << std::endl;
			msg.header.size = msg.size();

			std::string text(msg.body.begin(), msg.body.end());
			//std::cout << "SENDING: " << text << std::endl;
			return msg;
		}

		template <typename DataType>
		friend message<T>& operator >> (message<T>& msg, DataType& data) {
			static_assert(std::is_standard_layout<DataType>::value, "Data is too complex to be extracted from vector");

			size_t i = msg.body.size() - sizeof(data);

			std::memcpy(&data, msg.body.data() + i, sizeof(data));

			msg.body.resize(i);
			msg.header.size = msg.size();

			return msg;
		}

		friend message<T>& operator >> (message<T>& msg, std::string& data) {
			data.resize(msg.size());
			std::memcpy(data.data(), msg.body.data(), msg.size());

			msg.body.resize(0);
			msg.header.size = msg.size();

			return msg;
		}
	};

	template <typename T>
	class Connection;

	template <typename T>
	struct owned_message {
		std::shared_ptr<Connection<T>> remote = nullptr;
		message<T> msg;

		friend std::ostream& operator << (std::ostream& os, const owned_message<T>& msg) {
			os << msg.msg;
			return os;
		}
	};

}
