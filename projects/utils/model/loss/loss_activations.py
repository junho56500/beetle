1. Cross-Entropy Loss (CE)
Cross-Entropy Loss (CE) is the foundation of classification models. It measures the 
difference between the true probability distribution 
(the one-hot encoded label, y) and the predicted probability distribution (y^).

Goal: To maximize the likelihood of the correct class.
Strength: Works extremely well on balanced datasets with clean labels, as it severely 
penalizes confident wrong predictions.
Weakness (The Problem it Creates): It treats all classification errors equally. In a 
highly imbalanced dataset 
(e.g., 99% background, 1% objects), the 99% of easy, well-classified background examples 
contribute the vast majority of the total loss, 
drowning out the contribution of the important, rare objects.

2. Focal Loss (FL)
Focal Loss was introduced to solve the challenge of extreme foreground-background class 
imbalance in tasks like object detection.
Mechanism: It modifies the standard Cross-Entropy loss by adding a modulating factor to 
down-weight the loss contribution of easy examples.
FL = −αt​(1− p^t)^γ log(p^t)

Modulating Factor ((1−p^t)^γ):
If an example is easy (p^t is high, e.g., 0.99), then (1− p^t)^γ becomes close to 0, 
and the loss for that sample is significantly reduced.
If an example is hard (p^t is low, e.g., 0.1), then (1− p^t)^γ remains close to 1, and
 the loss contribution is large.

Strength: Forces the model to focus on hard, misclassified examples and ignore the sea 
of easy background examples, leading to higher performance on highly imbalanced tasks.
Weakness: Requires tuning of the focus parameter γ (usually 2.0) and the balancing 
factor α.

3. Symmetric Cross-Entropy Loss (SCE)
Symmetric Cross-Entropy Loss was designed to provide robustness against noisy labels 
(when the ground truth label y itself is wrong).

Mechanism: It combines the standard Cross-Entropy (CE) term with a Reverse Cross-Entropy
 (RCE) 
term: SCE=CE(y,y^)+λ⋅RCE(y^,y)
CE Term (CE(y,y^)): Focuses on convergence. It drives the prediction y^ toward the 
label y.

RCE Term (RCE(y^,y)): Provides robustness. The RCE term measures the CE from the 
prediction y^ to the label y. 
In simple terms, it acts to penalize the model less when its high confidence (y^≈1) 
is contradicted by a potentially noisy label (y). 
This prevents the model from being overconfident in an incorrect (noisy) label.

Strength: Excellent for scenarios where the training labels are known to be corrupted
 or noisy, preventing the model from overfitting to bad data.
Weakness: The RCE term can sometimes slow down the convergence speed compared to pure
 CE, and it requires tuning the balancing factor λ.

