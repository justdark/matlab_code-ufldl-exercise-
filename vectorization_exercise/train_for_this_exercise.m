visibleSize = 28*28;
hiddenSize = 196;
sparsityParam = 0.1;
lambda = 3e-3;
beta = 3;

images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
patches=zeros(visibleSize,10000);
for i=1:10000
    patches(:,i)=reshape(images(:,i),visibleSize,1);
end
beta
theta = initializeParameters(hiddenSize, visibleSize);
%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);display_network(W1', 12); 

print -djpeg weights.jpg   % save the visualization to a file 
