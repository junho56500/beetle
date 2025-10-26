#include <iostream>
#include <functional>


// Student.h
class Student {
private:
    std::string name;
    std::string rollNo;
public:
    // Getter/Setter methods
    void setName(const std::string& name);
    std::string getName() const;
    void setRollNo(const std::string& rollNo);
    std::string getRollNo() const;
};

// StudentView.h
class StudentView {
public:
    // Method to display student details
    void printStudentDetails(const std::string& studentName, const std::string& studentRollNo) const;
};

// StudentController.h
class StudentController {
private:
    Student* model;
    StudentView* view;
public:
    StudentController(Student* m, StudentView* v) : model(m), view(v) {}
    
    // Methods to interact with the Model
    void setStudentName(const std::string& name) { model->setName(name); }
    // ... other setters/getters
    
    // Method to tell the View to display the current Model data
    void updateView() {
        view->printStudentDetails(model->getName(), model->getRollNo());
    }
};

// main.cpp
int main() {
    // 1. Fetch data for the Model
    Student* initialStudent = new Student();
    initialStudent->setName("Robert");
    initialStudent->setRollNo("10");

    // 2. Create View and Controller
    StudentView* view = new StudentView();
    StudentController* controller = new StudentController(initialStudent, view);

    // 3. Display initial data (Controller tells View to use Model)
    controller->updateView(); 
    // Output: Name: Robert, Roll No: 10

    // 4. Update the data (Controller updates Model)
    controller->setStudentName("John");

    // 5. Display updated data
    controller->updateView();
    // Output: Name: John, Roll No: 10
    
    // Clean up
    delete initialStudent;
    delete view;
    delete controller;
    
    return 0;
}