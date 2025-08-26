function [senseMaps, eigenVal] = nullspace_vectors_G_matrix_2D(kCal, N1, N2, G, patchSize, ...
    varargin)

% Function that calculates the nullspace vectors for each G(x) matrix. These
% vectors correspond to sensitivity maps at the x location.
%
% Input parameters:
%   --kCal:         N1_cal x N2_cal x Nc block of calibration data, where
%                   N1_cal and N2_cal are the dimensions of a rectangular
%                   block of Nyquist-sampled k-space, and Nc is the number
%                   of channels in the array.
%
%   --N1, N2:       The desired dimensions of the output sensitivity
%                   matrices.
%
%   --G:            N1_g x N2_g x Nc x Nc array where G[i,j,:,:]
%                   corresponds to the G matrix at the (i,j) spatial
%                   location.
%
%   --patchSize:    Number of elements in the kernel used to calculate the
%                   nullspace vectors of the C matrix.
%
%   --PowerIteration_G_nullspace_vectors: Binary variable. 0 = nullspace
%                   vectors of the G matrices are calculated using SVD.
%                   1 = nullspace vectors of the G matrices are calculated
p = inputParser;

p.addRequired('kCal', @(x) isnumeric(x) && ndims(x) == 3);
p.addRequired('N1', @(x) isnumeric(x) && isscalar(x));
p.addRequired('N2', @(x) isnumeric(x) && isscalar(x));
p.addRequired('G', @(x) isnumeric(x) && ndims(x) == 4);
p.addRequired('patchSize', @(x) isnumeric(x) && isscalar(x));

p.addParameter('PowerIteration_G_nullspace_vectors', 1, @(x) isnumeric(x) && isscalar(x));
p.addParameter('M', 30, @(x) isnumeric(x) && isscalar(x));
p.addParameter('PowerIteration_flag_convergence', 1, @(x) isnumeric(x) && isscalar(x));
p.addParameter('PowerIteration_flag_auto', 0, @(x) isnumeric(x) && isscalar(x));
p.addParameter('FFT_interpolation', 1, @(x) isnumeric(x) && isscalar(x));
p.addParameter('gauss_win_param', 100, @(x) isnumeric(x) && isscalar(x));
p.addParameter('verbose', 1, @(x) isnumeric(x) && isscalar(x));

if isempty(varargin)
    parse(p, kCal, N1, N2, G, patchSize);
else
    parse(p, kCal, N1, N2, G, patchSize, varargin{:});
end

% Normalize optional flags to scalar logicals (don't modify p.Results which is read-only)
flagAuto = logical(p.Results.PowerIteration_flag_auto);
flagConv = logical(p.Results.PowerIteration_flag_convergence);
if flagAuto
    flagConv = 0; % auto mode suppresses convergence error check
end

N1_g = size(p.Results.G, 1);
N2_g = size(p.Results.G, 2);
Nc = size(p.Results.G, 3);

senseMaps = zeros(N1_g, N2_g, Nc);

G = reshape(permute(p.Results.G, [3 4 1 2]), [Nc Nc (N1_g * N2_g)]);

if p.Results.PowerIteration_G_nullspace_vectors == 0
    [~, eigenVal, Vpage] = pagesvd(G, 'econ', 'vector');
    eigenVal = reshape(permute(eigenVal, [3 1 2]), [N1_g, N2_g, Nc]);
    senseMaps = reshape(permute(Vpage(:, end, :), [3 1 2]), [N1_g N2_g Nc]);
    clear G
    eigenVal = eigenVal / p.Results.patchSize;
else
    G = G / p.Results.patchSize;
    G_null = repmat(eye(Nc), [1 1 (N1_g * N2_g)]);
    G_null = G_null - G;
    clear G

    if ~flagConv && ~flagAuto
        S = randn(Nc, 1) + 1i * randn(Nc, 1);
        S = repmat(S, [1 1 N1_g * N2_g]);
        S(:, 1, :) = S(:, 1, :) ./ pagenorm(S(:, 1, :));

        for m = 1:p.Results.M
            S = pagemtimes(G_null, S);
            S(:, 1, :) = S(:, 1, :) ./ pagenorm(S(:, 1, :));
            if m == p.Results.M
                E = pagemtimes(S, 'ctranspose', pagemtimes(G_null, S), 'none');
            end
        end

        clear G_null
        senseMaps = reshape(permute(squeeze(S), [2 1]), [N1_g, N2_g, Nc]);
        eigenVal = reshape(permute(E, [3 1 2]), [N1_g, N2_g]);
        eigenVal = 1 - eigenVal;
    end

    if flagConv || flagAuto
        S = randn(Nc, 1) + 1i * randn(Nc, 1);
        S2 = randn(Nc, 1) + 1i * randn(Nc, 1);
        S(:, 1, :) = S(:, 1, :) ./ pagenorm(S(:, 1, :));
        S2(:, 1, :) = S2(:, 1, :) ./ pagenorm(S2(:, 1, :));

        for m = 1:p.Results.M
            S = pagemtimes(G_null, S);
            S2 = pagemtimes(G_null, S2);
            S(:, 1, :) = S(:, 1, :) ./ pagenorm(S(:, 1, :));
            inner_prod = pagemtimes(S(:, 1, :), 'ctranspose', S2(:, 1, :), 'none');
            S2(:, 1, :) = S2(:, 1, :) - inner_prod .* S(:, 1, :);
            S2(:, 1, :) = S2(:, 1, :) ./ pagenorm(S2(:, 1, :));
            if m == p.Results.M
                E = pagemtimes(S, 'ctranspose', pagemtimes(G_null, S), 'none');
                E2 = pagemtimes(S2, 'ctranspose', pagemtimes(G_null, S2), 'none');
            end
        end

        eigen1 = reshape(permute(E, [3 1 2]), [N1_g, N2_g]);
        eigen2 = reshape(permute(E2, [3 1 2]), [N1_g, N2_g]);

        if p.Results.FFT_interpolation == 0
            eigenVal = 1 - eigen1;
            threshold_mask = 0.075;
            support_mask = zeros(size(eigenVal));
            support_mask(find(eigenVal < threshold_mask)) = 1;
            ratioEig = (eigen2 ./ eigen1) .^ p.Results.M;
            ratio_small = support_mask .* ratioEig;
            th_ratio = 0.008;
            ratio_small(find(ratio_small <= th_ratio)) = 0;
            ratio_small(find(ratio_small > th_ratio)) = 1;
            flag_convergence_PI = sum(ratio_small(:)) > 0;

            if flag_convergence_PI == 1 && flagConv
                error(['Power Iteration might have not converged for some voxels within the support after the ' int2str(p.Results.M) ...
                    ' iterations indicated by the user. Increasing the number of iterations is recommended. You can ignore this error by setting PowerIteration_flag_convergence = 0. ' ...
                    'The number of needed iterations for convergence can be found automatically by setting PowerIteration_flag_auto = 1. '])
            end

            if flag_convergence_PI == 0
                if p.Results.verbose == 1
                    disp(['Most likely Power Iteration has converged for all the voxels within the support after the ' int2str(p.Results.M) ' iterations indicated by the user.'])
                end
            end

            if flagAuto && flag_convergence_PI == 1
                if p.Results.verbose == 1
                    warning('off', 'backtrace')
                    warning(['Power Iteration might have not converged for some voxels within the support after the ' int2str(p.Results.M) ...
                        ' iterations indicated by the user. The number of iterations for the convergence of Power Iteration will be found automatically. You can turn off this option by setting PowerIteration_flag_auto = 0. '])
                end
                M_auto = p.Results.M + 1;
                while (flag_convergence_PI == 1)
                    S = pagemtimes(G_null, S);
                    S2 = pagemtimes(G_null, S2);
                    S(:, 1, :) = S(:, 1, :) ./ pagenorm(S(:, 1, :));
                    inner_prod = pagemtimes(S(:, 1, :), 'ctranspose', S2(:, 1, :), 'none');
                    S2(:, 1, :) = S2(:, 1, :) - inner_prod .* S(:, 1, :);
                    S2(:, 1, :) = S2(:, 1, :) ./ pagenorm(S2(:, 1, :));
                    E = pagemtimes(S, 'ctranspose', pagemtimes(G_null, S), 'none');
                    E2 = pagemtimes(S2, 'ctranspose', pagemtimes(G_null, S2), 'none');
                    eigen1 = reshape(permute(E, [3 1 2]), [N1_g, N2_g]);
                    eigen2 = reshape(permute(E2, [3 1 2]), [N1_g, N2_g]);
                    eigenVal = 1 - eigen1;
                    support_mask = zeros(size(eigenVal));
                    support_mask(find(eigenVal < threshold_mask)) = 1;
                    ratioEig = (eigen2 ./ eigen1) .^ M_auto;
                    ratio_small = support_mask .* ratioEig;
                    ratio_small(find(ratio_small <= th_ratio)) = 0;
                    ratio_small(find(ratio_small > th_ratio)) = 1;
                    flag_convergence_PI = sum(ratio_small(:)) > 0;
                    M_auto = M_auto + 1;
                end

                if p.Results.verbose == 1
                    disp(['Most likely Power Iteration has converged for all the voxels within the support. ' int2str(M_auto) ' iterations were needed.'])
                end
            end

            eigenVal = abs(eigenVal);
        end

        senseMaps = reshape(permute(squeeze(S), [2 1]), [N1_g, N2_g, Nc]);
    end
end

% ==== FFT-based interpolation ====
if p.Results.FFT_interpolation == 1
    [N1_cal, N2_cal, ~] = size(kCal);
    w_sm = (0.54 - 0.46 * cos(2 * pi * ((0:(N1_g - 1)) / (N1_g - 1))))';
    w_sm2 = (0.54 - 0.46 * cos(2 * pi * ((0:(N2_g - 1)) / (N2_g - 1))))';
    w_sm = w_sm * w_sm2';
    w_sm = repmat(w_sm, [1 1 Nc]);

    if p.Results.PowerIteration_G_nullspace_vectors == 1 && (flagConv || flagAuto)
        auxVal = 1 - eigen1;
        S_aux = fftshift(fft2(ifftshift(auxVal)));
        S_aux = S_aux .* w_sm(:, :, end);
        auxVal_us = abs(fftshift(fftshift(ifft2(S_aux, N1, N2), 1), 2));
        eigenVal = auxVal_us ./ max(auxVal_us(:));
        threshold_mask = 0.075;
        support_mask = (eigenVal < threshold_mask);

        S1_aux = fftshift(fft2(ifftshift(eigen1)));
        S1_aux = S1_aux .* w_sm(:, :, end);
        eigen1_us = abs(fftshift(fftshift(ifft2(S1_aux, N1, N2), 1), 2));

        S2_aux = fftshift(fft2(ifftshift(eigen2)));
        S2_aux = S2_aux .* w_sm(:, :, end);
        eigen2_us = abs(fftshift(fftshift(ifft2(S2_aux, N1, N2), 1), 2));

        ratioEig = (eigen2_us ./ eigen1_us) .^ p.Results.M;
        ratio_small = support_mask .* ratioEig;
        th_ratio = 0.008;
        ratio_small(ratio_small <= th_ratio) = 0;
        ratio_small(ratio_small > th_ratio) = 1;
        flag_convergence_PI = any(ratio_small(:));

        if flag_convergence_PI == 1 && flagConv
            error(['Power Iteration might have not converged for some voxels within the support after the ' int2str(p.Results.M) ...
                ' iterations indicated by the user. Increasing the number of iterations is recommended. You can ignore this error by setting PowerIteration_flag_convergence = 0. ' ...
                'The number of needed iterations for convergence can be found automatically by setting PowerIteration_flag_auto = 1. '])
        end

        if flag_convergence_PI == 0
            if p.Results.verbose == 1
                disp(['Most likely Power Iteration has converged for all the voxels within the support after the ' int2str(p.Results.M) ' iterations indicated by the user.'])
            end
        end

        if flagAuto && flag_convergence_PI == 1
            if p.Results.verbose == 1
                warning('off', 'backtrace')
                warning(['Power Iteration might have not converged for some voxels within the support after the ' int2str(p.Results.M) ...
                    ' iterations indicated by the user. The number of iterations for the convergence of Power Iteration will be found automatically. You can turn off this option by setting PowerIteration_flag_auto = 0.'])
            end
            M_auto = p.Results.M + 1;
            while (flag_convergence_PI == 1)
                S = pagemtimes(G_null, S);
                S2 = pagemtimes(G_null, S2);
                S(:, 1, :) = S(:, 1, :) ./ pagenorm(S(:, 1, :));
                inner_prod = pagemtimes(S(:, 1, :), 'ctranspose', S2(:, 1, :), 'none');
                S2(:, 1, :) = S2(:, 1, :) - inner_prod .* S(:, 1, :);
                S2(:, 1, :) = S2(:, 1, :) ./ pagenorm(S2(:, 1, :));
                E = pagemtimes(S, 'ctranspose', pagemtimes(G_null, S), 'none');
                E2 = pagemtimes(S2, 'ctranspose', pagemtimes(G_null, S2), 'none');
                eigen1 = reshape(permute(E, [3 1 2]), [N1_g, N2_g]);
                eigen2 = reshape(permute(E2, [3 1 2]), [N1_g, N2_g]);
                auxVal = 1 - eigen1;
                T = fftshift(fft2(ifftshift(auxVal)));
                T = T .* w_sm(:, :, end);
                aux_us = abs(fftshift(fftshift(ifft2(T, N1, N2), 1), 2));
                eigenVal = aux_us ./ max(aux_us(:));
                support_mask = (eigenVal < threshold_mask);
                T1 = fftshift(fft2(ifftshift(eigen1)));
                T1 = T1 .* w_sm(:, :, end);
                eigen1_us = abs(fftshift(fftshift(ifft2(T1, N1, N2), 1), 2));
                T2 = fftshift(fft2(ifftshift(eigen2)));
                T2 = T2 .* w_sm(:, :, end);
                eigen2_us = abs(fftshift(fftshift(ifft2(T2, N1, N2), 1), 2));
                ratioEig = (eigen2_us ./ eigen1_us) .^ M_auto;
                ratio_small = support_mask .* ratioEig;
                ratio_small(ratio_small <= th_ratio) = 0;
                ratio_small(ratio_small > th_ratio) = 1;
                flag_convergence_PI = any(ratio_small(:));
                M_auto = M_auto + 1;
            end

            if p.Results.verbose == 1
                disp(['Most likely Power Iteration has converged for all the voxels within the support. ' int2str(M_auto) ' iterations were needed.'])
            end
        end
    end

    if p.Results.PowerIteration_G_nullspace_vectors == 1 && ~flagConv && ~flagAuto
        T = fftshift(fft2(ifftshift(eigenVal)));
        T = T .* w_sm(:, :, end);
        eigenVal = abs(fftshift(fftshift(ifft2(T, p.Results.N1, p.Results.N2), 1), 2));
        eigenVal = eigenVal / max(eigenVal(:));
    end

    if p.Results.PowerIteration_G_nullspace_vectors == 0
        T = fftshift(fft2(ifftshift(eigenVal)));
        T = T .* w_sm(:, :, end);
        eigenVal = abs(fftshift(fftshift(ifft2(T, p.Results.N1, p.Results.N2), 1), 2));
        eigenVal = eigenVal / max(eigenVal(:));
    end

    % FFT-based interpolation of sensitivity maps
    win1 = gausswin(N1_g, p.Results.gauss_win_param);
    win2 = gausswin(N2_g, p.Results.gauss_win_param)';
    apod2D = win1 * win2;

    imLowRes_cal = zeros(N1_g, N2_g, Nc, 'like', senseMaps);
    cx = ceil(N1_g / 2) + utils.even_pisco(N1_g / 2);
    cy = ceil(N2_g / 2) + utils.even_pisco(N2_g / 2);
    hx = floor(N1_cal / 2);
    hy = floor(N2_cal / 2);
    rowIdx = cx + (-hx: hx - utils.even_pisco(N1_cal / 2));
    colIdx = cy + (-hy: hy - utils.even_pisco(N2_cal / 2));
    imLowRes_cal(rowIdx, colIdx, :) = p.Results.kCal;

    tmp = imLowRes_cal .* apod2D;
    tmp = ifftshift(tmp);
    imLowRes_cal = fftshift(ifft2(tmp));

    num = sum(conj(senseMaps) .* imLowRes_cal, 3);
    den = sum(abs(senseMaps) .^ 2, 3);
    cim = num ./ den;
    phase_cim = exp(1i * angle(cim));
    senseMaps = senseMaps .* phase_cim;

    S = fftshift(fft2(ifftshift(senseMaps)));
    S = S .* w_sm;
    senseMaps = fftshift(fftshift(ifft2(S, p.Results.N1, p.Results.N2), 1), 2);
end

end