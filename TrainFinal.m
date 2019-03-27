%%
% Note: This file contains the code which was used to find the best
% possible hyperparameters 'KernelScale' (k) and 'BoxConstraint' (C). The
% process is discussed in detail in the file 'Exercise5.m'.

%% Loading and preprocessing the data
load mnist.mat;
img = cell2mat(cellfun(@(x) reshape(x,[28*28,1]),img,'UniformOutput',false))';
img_test = cell2mat(cellfun(@(x) reshape(x,[28*28,1]),img_test,'UniformOutput',false))';
% Normalizing each column:
img_test = cell2mat(cellfun(@(x) mat2gray(x), num2cell(img_test,1),'UniformOutput',false));
img = cell2mat(cellfun(@(x) mat2gray(x), num2cell(img,1),'UniformOutput',false));
%%
% Using a subset to train and validate
X = img(1:100,:);
y = labels(1:100,:);
Xt = img(end-99:end,:);
yt = labels(end-99:end,:);


%%
% Trying all possible k,C duplet from 1 to 100 to which gives the least
% cost
for k = 1:100
    for C = 1:100
        fprintf(strcat(num2str(k),32,num2str(C),'\n\r'));
        params = templateSVM('KernelFunction','rbf', 'KernelScale',k,'BoxConstraint', C);
        mdl = fitcecoc(X,y,'Learners', params, 'Coding', 'onevsall');
        result(k,C) = sum(predict(mdl, Xt) ~= yt);
    end
end

% I and J are column vectors containing (i,j) pairs that give the least
% cost.
[I,J] = find(result == min(result(:)));
%%
% Training for each (k,C) duplet on a bigger (and entirely different)
% dataset for trainign and validation

I(:,2) = J; % Combining I and J into one
X = img(101:1000,:);
y = labels(101:1000,:);
Xt = img(end-999:end-100,:);
yt = labels(end-999:end-100,:);

for i = 1:size(I,1)
    k = I(i,1); C = I(i,2);
    fprintf(strcat(num2str(k),32,num2str(C),'\n\r'));
    params = templateSVM('KernelFunction','rbf', 'KernelScale',k,'BoxConstraint', C);
    mdl = fitcecoc(X,y,'Learners', params, 'Coding', 'onevsall');
    result2(k,C) = sum(predict(mdl, Xt) ~= yt);
end

% result2.mat was narrowed down to have just one pair of k,C values, that
% is (6,1).

save research.mat result result2;
