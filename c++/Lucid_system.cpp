#include <iostream>
#include <memory>
#include <sched.h>
#include <vector>

using namespace std;

// M1, M2, M3, M4, M5
// M2 dep M1
// M3 dep M1
// M4 dep M2 & M3
// M5 dep M4

// M1 -> M2 -> M4 ->M5
//    -> M3

class Scheduler
{
public:
  void getInput(string Task, string depTask)
  {
    bool isExist = false;
    for (int i = 0; i < tasks.size(); i++)
    {
      for (int j = 0; j < tasks[0].size(); j++)
      {
        if (tasks[i][j] == depTask)
        {
          tasks[i].push_back(Task);
          isExist = true;
        }
        else if(tasks[i][j] == Task)
        {
          isExist = true;
          break;
        }
      }
    }
    if (!isExist)
    {
      tasks[0].push_back(depTask);
      tasks[0].push_back(Task);
    }


  };

  vector<vector<string>> getResult()
  {
    return tasks;
  }

private:
  vector<vector<string>> tasks;

};

// To execute C++, please define "int main()"
int main() {
  Scheduler sch;

  sch.getInput("M2", "M1");
  sch.getInput("M3", "M1");
  sch.getInput("M4", "M2");
  sch.getInput("M4", "M3");
  sch.getInput("M5", "M4");

  auto result = sch.getResult();

  
  for (const auto& vec: result)
  {
    for (const auto& i: vec)
    {
      cout << i << " -> ";
    }
  }

  return 0;
}
