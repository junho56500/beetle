#include "stdoio.h"

class MyClass {
    public:
        MyClass() = default; // 디폴트 생성자 사용
        MyClass(const MyClass& other) = default; // 디폴트 복사 생성자 사용
        MyClass& operator=(const MyClass& other) = default; // 디폴트 대입 연산자 사용
};

class MyClass {
    public:
        MyClass() = delete; // 생성자 사용 불가
        MyClass(const MyClass& other) = delete; // 복사 생성자 사용 불가
        MyClass& operator=(const MyClass& other) = delete; // 대입 연산자 사용 불가
};

class Parent {
    public:
        virtual void someFunction() {
            // ...
        }
};

class Child : public Parent {
    public:
        void someFunction() override { // 부모 클래스의 someFunction 재정의
            // ...
        }
};


class MyClass {
    public:
        explicit MyClass(int value) { // MyClass를 int로 변환할 수 없다.
            // ...
        }
};

int main() {
    MyClass myObject(10); // 명시적 형 변환 가능
    // MyClass myObject = 10; // 오류: 암시적 형 변환 불가
    return 0;
}