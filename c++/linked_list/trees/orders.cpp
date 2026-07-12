#include <iostream>
using namespace std;

struct Node {
    int data;
    Node* left;
    Node* right;
    Node(int val) : data(val), left(nullptr), right(nullptr) {}
};

// 1. 전위 순회 (Root -> Left -> Right)
void preOrder(Node* node) {
    if (node == nullptr) return;
    cout << node->data << " "; // 루트 방문
    preOrder(node->left);      // 왼쪽 서브트리
    preOrder(node->right);     // 오른쪽 서브트리
}

// 2. 중위 순회 (Left -> Root -> Right)
void inOrder(Node* node) {
    if (node == nullptr) return;
    inOrder(node->left);       // 왼쪽 서브트리
    cout << node->data << " "; // 루트 방문
    inOrder(node->right);      // 오른쪽 서브트리
}

// 3. 후위 순회 (Left -> Right -> Root)
void postOrder(Node* node) {
    if (node == nullptr) return;
    postOrder(node->left);      // 왼쪽 서브트리
    postOrder(node->right);     // 오른쪽 서브트리
    cout << node->data << " ";  // 루트 방문
}

int main() {
    /* 아래와 같은 모양의 트리 생성
                1
               / \
              2   3
             / \
            4   5
    */
    Node* root = new Node(1);
    root->left = new Node(2);
    root->right = new Node(3);
    root->left->left = new Node(4);
    root->left->right = new Node(5);

    cout << "전위 순회 (Pre-order): ";
    preOrder(root);
    cout << "\n";

    cout << "중위 순회 (In-order): ";
    inOrder(root);
    cout << "\n";

    cout << "후위 순회 (Post-order): ";
    postOrder(root);
    cout << "\n";

    // 메모리 해제 로직은 생략되었습니다. (실제 사용시 동적할당 해제 필요)
    return 0;
}

// 전위 순회 (Pre-order): 1 2 4 5 3 
// 중위 순회 (In-order): 4 2 5 1 3 
// 후위 순회 (Post-order): 4 5 2 3 1