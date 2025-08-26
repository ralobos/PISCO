function U = nullspace_vectors_C_matrix_2D(kCal, varargin)

% Function that returns the nullspace vectors of the C matrix.
%
% Input parameters:
%   --kCal:                      N1_cal x N2_cal x Nc block of calibration data, where
%                                N1_cal and N2_cal are the dimensions of a rectangular
%                                block of Nyquist-sampled k-space, and Nc is the number
%                                of channels in the array.
%
%   --tau:                       Parameter (in Nyquist units) that determines the size of
%                                the k-space kernel. For a rectangular kernel, the size
%                                corresponds to (2*tau+1) x (2*tau+1). For an ellipsoidal
%                                kernel, it corresponds to the radius of the associated
%                                neighborhood. Default: 3.
%
%   --threshold:                 Specifies how small a singular value needs to be
%                                (relative to the maximum singular value) before its
%                                associated singular vector is considered to be in the
%                                nullspace of the C-matrix. Default: 0.05.
%
%   --kernel_shape:              Binary variable. 0 = rectangular kernel, 1 = ellipsoidal
%                                kernel. Default: 1.
%
%   --FFT_nullspace_C_calculation: Binary variable. 0 = nullspace vectors of
%                                the C matrix are calculated from C'*C by calculating C
%                                first. 1 = nullspace vectors of the C matrix are calculated
%                                from C'*C, which is calculated directly using an FFT-based
%                                approach. Default: 1.
%
%   --sketched_SVD:              Binary variable. 1 = sketched SVD is used to calculate
%                                a basis for the nullspace of the C matrix. Default: 1.
%
%   --sketch_dim:                Dimension of the sketch matrix used to calculate a basis
%                                for the nullspace of the C matrix using a sketched SVD.
%                                Only used if sketched_SVD is enabled. Default: 500.
%
%   --visualize_C_matrix_sv:     Binary variable. 1 = Singular values of the C matrix are displayed.
%                                Default: 0. 
%                                Note: If sketched_SVD = 1 and if the curve of the singular values flattens out,
%                                it suggests that the sketch dimension is appropriate for the data.
%
% Output parameters:
%   --U:                         Matrix whose columns correspond to the nullspace vectors
%                                of the C matrix.

p = inputParser;

p.addRequired('kCal', @(x) isnumeric(x) && ndims(x) == 3);

p.addParameter('tau', @(x) isnumeric(x) && isscalar(x));
p.addParameter('threshold', @(x) isnumeric(x) && isscalar(x));
p.addParameter('kernel_shape', @(x) isnumeric(x) && isscalar(x));
p.addParameter('FFT_nullspace_C_calculation', @(x) isnumeric(x) && isscalar(x));
p.addParameter('pad', 1, @(x) isnumeric(x) && isscalar(x) && (x == 0 || x == 1));
p.addParameter('sketched_SVD', @(x) isnumeric(x) && isscalar(x));
p.addParameter('sketch_dim', @(x) isnumeric(x) && isscalar(x));
p.addParameter('visualize_C_matrix_sv', @(x) isnumeric(x) && isscalar(x));

if isempty(varargin)
    parse(p, kCal);
else
    parse(p, kCal, varargin{:});
end

if p.Results.FFT_nullspace_C_calculation == 0 

    opts_C_matrix = struct( ...
    'tau', p.Results.tau,...
    'kernel_shape', p.Results.kernel_shape...
    );

    fn = fieldnames(opts_C_matrix);
    fv = struct2cell(opts_C_matrix);
    nv = [fn.'; fv.'];
    nv = nv(:).';

    C = utils.C_matrix_2D(p.Results.kCal, nv{:});

    ChC = C'*C;
    clear C
    
else

    opts_ChC_matrix = struct( ...
    'tau', p.Results.tau,...
    'pad', p.Results.pad,...
    'kernel_shape', p.Results.kernel_shape...
    );

    fn = fieldnames(opts_ChC_matrix);
    fv = struct2cell(opts_ChC_matrix);
    nv = [fn.'; fv.'];
    nv = nv(:).';

    ChC = utils.ChC_FFT_convolutions_2D(p.Results.kCal, nv{:});

end

if p.Results.sketched_SVD == 0

    [~,Sc,U] = svd(ChC,'econ');
    clear ChC
    sing = diag(Sc);
    clear Sc
    
    sing = sqrt(sing);
    sing  = sing/sing(1);

    if p.Results.visualize_C_matrix_sv == 1

        % Visualize singular values of the C matrix
        figure;
        plot(sing, 'o-');
        title('Singular values of the C matrix');
        grid on;
        xlim([1 numel(sing)]);
        ylim([0 1]);
        xlabel('Index');
        ylabel('Singular value');

    end

    Nvect = find(sing >= p.Results.threshold*sing(1),1,'last');
    clear sing
    U = U(:, Nvect+1:end); 

else

    %Sketching

    [~, N2c] = size(ChC);
    Sk = (1/sqrt(p.Results.sketch_dim))*randn(p.Results.sketch_dim, N2c) + (1/sqrt(p.Results.sketch_dim))*1i*randn(p.Results.sketch_dim, N2c);
    C = Sk*ChC;
    [~, sing, Vf] = svd(C, 'econ', 'vector');

    sing = sqrt(sing);
    sing  = sing/sing(1);

    if p.Results.visualize_C_matrix_sv == 1

        % Visualize singular values of the C matrix
        figure;
        plot(sing, 'o-');
        title('Singular values of the C matrix (sketched SVD)');
        grid on;
        xlim([1 numel(sing)]);
        ylim([0 1]);
        xlabel('Index');
        ylabel('Singular value');

    end

    rank_C = find(sing >= p.Results.threshold*sing(1),1,'last');
    clear sing

    U = Vf(:,1:rank_C);
    clear Vf
    
end
end