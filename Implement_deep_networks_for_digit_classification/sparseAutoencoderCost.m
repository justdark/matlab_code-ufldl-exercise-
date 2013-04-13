function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64xmm matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
mm=size(data,2);

z2=(W1*data)+repmat(b1,1,mm);
a2=sigmoid(z2);
z3=(W2*a2)+repmat(b2,1,mm);
a3=sigmoid(z3);

rho=sum(a2,2)./mm;

cost=1/2/mm*sum(sum((a3-data).^2))+lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2)))+beta*sum(kldiverge(rho,sparsityParam));
sparityy=(-sparsityParam./rho+(1-sparsityParam)./(1-rho));

 x=data();
 delta3=-(x-a3).*sigmoidGradient(z3);
 delta2=(W2'*delta3+repmat(beta*sparityy,1,mm)).*a2.*(1-a2);
 W1grad=W1grad+delta2*x';
 W2grad=W2grad+delta3*a2';
 b1grad=b1grad+sum(delta2,2);
 b2grad=b2grad+sum(delta3,2);

b1grad=b1grad/mm;
b2grad=b2grad/mm;
W1grad=W1grad/mm+lambda*W1;
W2grad=W2grad/mm+lambda*W2;









%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
function kld = kldiverge(x,sparsityParam)
    kld =sparsityParam.*log(sparsityParam./x)+(1-sparsityParam).*log((1-sparsityParam)./(1.-x));
end 

function g = sigmoidGradient(z)
g=zeros(size(z));
g=sigmoid(z).*(1-sigmoid(z));
end
