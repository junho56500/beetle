//Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

// Example 1:
// Input: root = [3,9,20,null,null,15,7]
// Output: [[3],[9,20],[15,7]]

// Example 2:
// Input: root = [1]
// Output: [[1]]

// Example 3:
// Input: root = []
// Output: []

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ret;
        
        if(root == nullptr)
        {
            return ret;
        }
        
        queue<TreeNode*> q;
        q.push(root);
        
        // map<vector<TreeNode*>, bool> visited;
        // ret.push_back(root->val);
        
        while(!q.empty())
        {
            int level_size = q.size();
            
            vector<int> cur_vals;
            
            for (int i=0; i<level_size; i++)
            {
                TreeNode* t;
                t = q.front();
                q.pop();
                cur_vals.push_back(t->val);
                if(t->left)
                    q.push(t->left);
                if(t->right)
                    q.push(t->right);
            }
            
            ret.push_back(cur_vals);
        }
        return ret;
    }
};