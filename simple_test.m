% Simple test to debug vectorization issue
clear; clc;

% Create simple test data
N1 = 8; N2 = 8; Nc = 2;
x = reshape(1:(N1*N2*Nc), N1*N2, Nc);  % Simple sequential data

tau = 1;
patchSize = (2*tau+1)^2;  % 3x3 = 9

% Original method (first iteration only)
i = tau+1; j = tau+1;  % center at (2,2)
[in1,in2] = meshgrid(-tau:tau,-tau:tau);
in1 = in1(:)'; in2 = in2(:)';

ind_orig = sub2ind([N1,N2], i+in1, j+in2);
result_orig = x(ind_orig, :);
result_orig_vec = result_orig(:);  % utils.vect equivalent

% Vectorized method  
I_all = repmat(i, 1, patchSize) + repmat(in1', 1, 1);
J_all = repmat(j, 1, patchSize) + repmat(in2', 1, 1);
ind_vec = sub2ind([N1,N2], I_all, J_all);
result_vec = x(ind_vec, :);
result_vec_flat = result_vec(:);

fprintf('Original indices: '); fprintf('%d ', ind_orig); fprintf('\n');
fprintf('Vectorized indices: '); fprintf('%d ', ind_vec); fprintf('\n');
fprintf('Indices equal? %s\n', mat2str(isequal(ind_orig(:), ind_vec(:))));

fprintf('Original result flat: '); fprintf('%.1f ', result_orig_vec(1:6)); fprintf('\n');
fprintf('Vectorized result flat: '); fprintf('%.1f ', result_vec_flat(1:6)); fprintf('\n');
fprintf('Results equal? %s\n', mat2str(isequal(result_orig_vec, result_vec_flat)));
