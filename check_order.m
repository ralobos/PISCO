% Check the order of centers from nested loops
clear; clc;

tau = 3;
N1 = 32; N2 = 32;

% even_pisco function (assuming it returns 0 for even, 1 for odd)
even_pisco = @(n) mod(n,2) == 0;

% Original nested loop order
i_centers = tau+1+even_pisco(N1):N1-tau;
j_centers = tau+1+even_pisco(N2):N2-tau;

fprintf('i_centers: %d to %d (%d values)\n', i_centers(1), i_centers(end), length(i_centers));
fprintf('j_centers: %d to %d (%d values)\n', j_centers(1), j_centers(end), length(j_centers));

% Method 1: Nested loop order (original)
centers_loop = [];
for i_val = i_centers
    for j_val = j_centers
        centers_loop = [centers_loop; i_val, j_val];
    end
end

% Method 2: Meshgrid order
[J_centers, I_centers] = meshgrid(j_centers, i_centers);
centers_mesh = [I_centers(:), J_centers(:)];

fprintf('First 5 centers from nested loops:\n');
for k = 1:5
    fprintf('  [%d, %d]\n', centers_loop(k,1), centers_loop(k,2));
end

fprintf('First 5 centers from meshgrid:\n');
for k = 1:5
    fprintf('  [%d, %d]\n', centers_mesh(k,1), centers_mesh(k,2));
end

fprintf('Centers match? %s\n', mat2str(isequal(centers_loop, centers_mesh)));
