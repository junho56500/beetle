
//Adapter (Wrapper)
// 1. The Target Interface (What the Client Expects)
// Target.h
#include <iostream>

class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
};


//2. The Adaptee Class (The Incompatible Existing Code)
// LegacyGeometryLibrary.h
class LegacyGeometryLibrary {
public:
    void render_line(int x1, int y1, int x2, int y2) const {
        std::cout << "Legacy Drawing: Rendering line from (" << x1 << ", " << y1 << ") to (" << x2 << ", " << y2 << ")\n";
    }
    // Assume other incompatible methods exist
};


//3. The Adapter Class (The Wrapper)
// LineAdapter.h
#include "Target.h"
#include "LegacyGeometryLibrary.h"

class LineAdapter : public Shape {
private:
    LegacyGeometryLibrary* adaptee;
    // Specific data needed for the legacy call, e.g., line coordinates
    int x1, y1, x2, y2; 

public:
    LineAdapter(int x1, int y1, int x2, int y2) 
        : x1(x1), y1(y1), x2(x2), y2(y2) {
        // The adapter might create or be given the adaptee instance
        adaptee = new LegacyGeometryLibrary();
    }

    ~LineAdapter() override {
        delete adaptee;
    }

    // Implementing the Target interface method
    void draw() const override {
        // The core of the Adapter: translating the Target call to the Adaptee call
        std::cout << "Adapter: Translating generic 'draw' to specific 'render_line'...\n";
        adaptee->render_line(x1, y1, x2, y2);
    }
};