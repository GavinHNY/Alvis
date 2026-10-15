#pragma once
#include "common.h"

namespace net {
	template<typename T>
	class tsQueue {
	public:
		tsQueue() = default;
		tsQueue(const tsQueue<T>&) = delete;
		virtual ~tsQueue() { clear(); }
	public:
		const T& front() {
			std::scoped_lock lock(muxQueue);
			return deqQueue.front();
		}
		const T& back() {
			std::scoped_lock lock(muxQueue);
			return deqQueue.back();
		}
		void push_back(const T& item) {
			std::scoped_lock lock(muxQueue);
			deqQueue.emplace_back(std::move(item));

			std::unique_lock<std::mutex> ul(muxBlocking);
			cvBlocking.notify_one();
		}
		void push_front(const T& item) {
			std::scoped_lock lock(muxQueue);
			deqQueue.emplace_front(std::move(item));

			std::unique_lock<std::mutex> ul(muxBlocking);
			cvBlocking.notify_one();
		}
		bool empty() {
			std::scoped_lock lock(muxQueue);
			return deqQueue.empty();
		}
		size_t count() {
			std::scoped_lock lock(muxQueue);
			return deqQueue.size();
		}
		void clear() {
			std::scoped_lock lock(muxQueue);
			deqQueue.clear();
		}

		T pop_front() {
			std::scoped_lock lock(muxQueue);
			auto t = std::move(deqQueue.front());
			deqQueue.pop_front();
			return t;
		}
		T pop_back() {
			std::scoped_lock lock(muxQueue);
			auto t = std::move(deqQueue.back());
			deqQueue.pop_back();
			return t;
		}

		void wait() {
			while (empty() && !m_bRunning) {
				std::unique_lock<std::mutex> ul(muxBlocking);
				cvBlocking.wait(ul);
			}
		}

		void notifyQuit() {
			std::unique_lock<std::mutex> ul(muxBlocking);
			m_bRunning = false;
			cvBlocking.notify_one();
		}


	protected:
		std::mutex muxQueue;
		std::deque<T> deqQueue;

		std::condition_variable cvBlocking;
		std::mutex muxBlocking;

		bool m_bRunning = true;
	};
}
