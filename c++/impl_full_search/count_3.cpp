#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

using namespace std;

// 0시 0분 0초부터 N시 59분 59초까지 3이 들어간 시각의 개수

int main()
{
    int n = 0;
    cin >> n;

    int cnt = 0;

    if(n>0)
    {
        for (int i=0; i<n; i++)
        {
            for (int j=0; j<=59; j++)
            {
  
                for (int k=0; k<=59; k++)
                {
                    if ((k==3) or (k%10 == 3) or (j==3) or (j%10 == 3) or (j==3) or (j%10 == 3))
                    {
                        cnt++;
                    }
                }
            }
        }
    }

    cout << cnt;

    return 0;
}
