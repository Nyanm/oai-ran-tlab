#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

template <typename T>
class NonBlockingQueue {
public:
    void push(const T& value) {
        std::unique_lock<std::mutex> lock(mtx_);
        queue_.push(value);
        cv_.notify_one(); // Or notify_all() for multiple consumers
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !queue_.empty(); });
        T value = queue_.front();
        queue_.pop();
        return value;
    }

    T pop_wait_for(int wait_us) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait_for(lock, std::chrono::microseconds(wait_us), [this] { return !queue_.empty(); });
        if (!queue_.empty()) {
            T value = queue_.front();
            queue_.pop();
            return value;
        }
        else {
            return nullptr;
        }

    }

private:
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
};