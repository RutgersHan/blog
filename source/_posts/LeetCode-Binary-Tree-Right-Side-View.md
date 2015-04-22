title: "[LeetCode]Binary Tree Right Side View"
date: 2015-04-15 00:49:25
categories: [LeetCode]
tags: 
---
Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.
<!--more-->

For example:
Given the following binary tree
```
   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---
```
You should return [1, 3, 4].

Solutions

1. Based on level order traversal and then print the last element in each level(iterative)
```cpp
/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> rightSideView(TreeNode *root) {
        vector<int>result;
        if(root==NULL)
        {
            return result;
        }
        queue<TreeNode*> Que_Nodes;
        Que_Nodes.push(root);
        Que_Nodes.push(NULL);
        while(!Que_Nodes.empty())
        {
            TreeNode* Current = Que_Nodes.front();
            Que_Nodes.pop();
            TreeNode* Next = Que_Nodes.front();
            if(Current->left!=NULL)
            {
                Que_Nodes.push(Current->left);
            }
            if(Current->right!=NULL)
            {
                Que_Nodes.push(Current->right);
            }
            if(Next==NULL && Que_Nodes.size()!=1)
            {
                result.push_back(Current->val);
                Que_Nodes.push(NULL);
                Que_Nodes.pop();
            }
            else if(Next==NULL && Que_Nodes.size()==1)
            {
                result.push_back(Current->val);
                break;
            }

        }
        
        
    }
};
```
2. Recursive Method, the size of the vector is a indicator of the max level now, if the current level is bigger than the max level, then the node should be put in the vector
```cpp
/**
* Definition for binary tree
* struct TreeNode {
*     int val;
*     TreeNode *left;
*     TreeNode *right;
*     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
* };
*/
class Solution {
public:
    vector<int> rightSideView(TreeNode *root) {
        vector<int> right_side;
        rightSide(root, right_side, 0);
        return right_side;
    }
    void rightSide(TreeNode *r, vector<int> &a, int i)
    {
        if (r == NULL)return;
        if (i == a.size())
            a.push_back(r->val);
        rightSide(r->right, a, i + 1);
        rightSide(r->left, a, i + 1);
    }
};
```