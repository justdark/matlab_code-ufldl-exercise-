function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));  %numClasses*M
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.


%size(M)
%size(repmat(sum(exp(M),1),numClasses,1))\
%h = exp(M)./repmat(sum(exp(M),1),numClasses,1); %numClasses*M
%h = bsxfun(@minus,h, max(h, [], 1));
%h(:,2)

M = theta*data;     % (numClasses,N)*(N,M)

M = bsxfun(@minus, M, max(M, [], 1));
h = exp(M);

h =  bsxfun(@rdivide, h, sum(h));
cost = -1/numCases*sum(sum(groundTruth.*log(h)))+lambda/2*sum(sum(theta.^2));
thetagrad = -1/numCases*((groundTruth-h)*data')+lambda*theta;%log(h)

%for i=1:numCases
 %       s=groundTruth(:,i).*log(h(:,i));
  %      cost=cost+sum(s);
%end
%cost=cost*(-1)/numCases+lambda/2*sum(sum(theta.^2));
%for i=1:numClasses
%    for j=1:numCases
%        %groundTruth(:,j)
%        %h(:,j)
%        k=((groundTruth(:,j)-h(:,j))*data(:,j)');
%        
%        thetagrad(i,:)=thetagrad(i,:)+k(i,:);
%    end
%     thetagrad(i,:)=-thetagrad(i,:)/numCases+lambda*theta(i,:);
%end

%thetagrad = -1/m*((y-h)*data')+lambda*theta;
%z = theta * data;
%z  = bsxfun(@minus, z, max(z, [], 1));
%h = exp(z);
%h = bsxfun(@rdivide, h, sum(h));
%cost = -1/m* sum( (y .* log(h))(:) ) + lambda/2*sum(theta(:).^2);

%data: N*M
%groundTruth: numClasses*M
%theta:  numClasses*N









% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

