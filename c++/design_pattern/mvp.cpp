#include <iostream>
#include <functional>

// 1. The Model: Manages the data and state (the counter value)
class CounterModel {
private:
    int count = 0;
    // A callback function to notify the Presenter/View of changes
    std::function<void(int)> change_callback;

public:
    void set_change_callback(std::function<void(int)> cb) {
        change_callback = std::move(cb);
    }

    int get_count() const {
        return count;
    }

    void increment() {
        count++;
        // Notify listeners (the Presenter) of the change
        if (change_callback) {
            change_callback(count);
        }
    }
};

// 2. The View Interface: What the Presenter needs to talk to the UI
class ICounterView {
public:
    virtual ~ICounterView() = default;
    
    // Method for the Presenter to tell the View what to display
    virtual void display_count(int count) = 0;
    
    // Method to allow the Presenter to register a handler for the 'Increment' action
    // In a real GUI, this would be triggered by a button click
    virtual void set_increment_handler(std::function<void()> handler) = 0;
};

// 3. The Presenter: The glue between Model and View
class CounterPresenter {
private:
    CounterModel& model;
    ICounterView* view;

public:
    CounterPresenter(CounterModel& m, ICounterView* v) : model(m), view(v) {
        // Wire up the Model's state change to a Presenter method
        model.set_change_callback(
            [this](int count) {
                this->on_model_count_changed(count);
            }
        );

        // Wire up the View's action to a Presenter method
        view->set_increment_handler(
            [this]() {
                this->on_increment_clicked();
            }
        );
        
        // Initial setup: display the current state
        view->display_count(model.get_count());
    }

private:
    // Handle events from the View (User Interaction)
    void on_increment_clicked() {
        // 1. Process the action (tell the Model to update)
        model.increment();
        // The Model will call back (on_model_count_changed) which will update the View
    }

    // Handle events from the Model (State Change)
    void on_model_count_changed(int new_count) {
        // 2. Update the View with the new data
        view->display_count(new_count);
    }
};

// 4. The Concrete View: The actual UI implementation
class ConsoleCounterView : public ICounterView {
private:
    // The handler to be called when the user "clicks" the increment button
    std::function<void()> increment_handler;

public:
    void display_count(int count) override {
        std::cout << "Display Updated -> Count: " << count << std::endl;
    }

    void set_increment_handler(std::function<void()> handler) override {
        increment_handler = std::move(handler);
    }

    // Simulating a button click event
    void simulate_increment_click() {
        std::cout << "\n[User Clicks Increment Button]";
        if (increment_handler) {
            increment_handler();
        }
    }
};


int main() {
    // 1. Create the Model and the View Implementation
    CounterModel model;
    ConsoleCounterView view;

    // 2. Create the Presenter and inject the Model and View
    CounterPresenter presenter(model, &view);

    // Initial state is displayed by the Presenter's constructor
    // Output: Display Updated -> Count: 0

    // Simulate user interaction
    view.simulate_increment_click();
    // Output: [User Clicks Increment Button]
    //         Display Updated -> Count: 1

    view.simulate_increment_click();
    // Output: [User Clicks Increment Button]
    //         Display Updated -> Count: 2

    return 0;
}