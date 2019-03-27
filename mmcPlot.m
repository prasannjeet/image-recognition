%% Maximal margin classifier
function beta = mmcPlot(X, y)
% Maximal margin classifier which computes and plots the hyperplane for 2D data

%% Solving the maximal margin classifier problem
p = 2;
H = eye(p+1);
H(1,1) = 0;
f = zeros(p+1,1);
b = -ones(size(X,1),1);
A = [-y -diag(y)*X];
beta = quadprog(H, f, A, b);

%% Find the support vectors
sv = (A*beta*(-1)<1.0001);

%% Normalizing beta
beta = beta/sqrt(beta(2:end)'*beta(2:end));

%% Find the distance from the support vectors to the hyperplane
% M should be the perpendicular distance from any support vector to the
% hyperplane
%
% *The Distance*
%
% If $ax+by+c=0$ is the equation of the hyperplane, and one of the support
% vector point is $(p,q)$, then the distance was calculated by:
%
% $$\frac{|ap+bq+c|}{\sqrt{a^2+b^2}}$$
%
% Implemented below:

% Choosing any single support vector (ssv) from all support vectos
ssv = X(sv,:); ssv = ssv(1,:);
% Now calculating the perpendicular distance
M = abs(beta(2)*ssv(1)+beta(3)*ssv(2)+beta(1))/((beta(2)^2+...
    beta(3)^2)^(1/2));

d = M/sin(acos(beta(2)/sqrt(beta(2:end)'*beta(2:end))));

%% Plotting the MMC, its slab and support vectors
l = linspace(min(X(:,1)),max(X(:,1)),100);
pos = (y==1);
neg = (y==-1);

clf;
hold on;
% Plotting the data
plot(X(pos,1), X(pos,2),'ko','MarkerFaceColor','y');
plot(X(neg,1), X(neg,2),'ko','MarkerFaceColor','r');

% Plotting the MMC
plot(l, -beta(2)/beta(3)*l + -beta(1)/beta(3)); 

% Plotting one slab
plot(l,  -beta(2)/beta(3)*l + -(beta(1))/beta(3) + d,'m--');

% Plotting the support vectors
plot(X(sv,1), X(sv,2), 'ko','MarkerSize',10);

% Plotting the other slab
plot(l,  -beta(2)/beta(3)*l + -(beta(1))/beta(3) - d,'m--');

legend('Positive','Negative','Model 1', 'Slab', 'Support vectors');
xlim([-3 3])
ylim([-3 3])
hold off;

end

