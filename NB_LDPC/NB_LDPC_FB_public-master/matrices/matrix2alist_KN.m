% matrix2alist_KN
% matrix to alisk with KN format

function matrix2alist_KN(H, gfOrder, output_filename)
% MATRIX2ALIST_KN  Convert a parity-check matrix to KN-format alist file.
%
%   matrix2alist_KN(H, gfOrder, output_filename)
%
%   Inputs:
%       H               - parity-check matrix of size MxN with GF elements
%       gfOrder         - order of the Galois Field (e.g., 16 for GF(16))
%       output_filename - target alist output file

    arguments
        H double
        gfOrder (1, 1) double {mustBePositive}
        output_filename (1, :) char
    end

    H_bin = (H ~= 0);
    [M, N] = size(H);

    dv = sum(H_bin, 1); % column weights
    dc = sum(H_bin, 2).'; % row weights as row vector

    cmax = max([dv, 0]);
    rmax = max([dc, 0]);

    fid = fopen(output_filename, 'w');
    if fid == -1
        error('matrix2alist_KN:IOError', 'Cannot open %s for writing.', output_filename);
    end
    cleaner = onCleanup(@() fclose(fid));

    fprintf(fid, '%d %d %d\n', N, M, gfOrder);
    fprintf(fid, '%d %d\n', cmax, rmax);

    fprintf(fid, '%d ', dv);
    fprintf(fid, '\n');

    fprintf(fid, '%d ', dc);
    fprintf(fid, '\n');

    % Column placeholder section (ignored by NB decoder, but required by format)
    for col = 1:N
        for k = 1:cmax
            fprintf(fid, '0 0');
            if k < cmax
                fprintf(fid, ' ');
            end
        end
        fprintf(fid, '\n');
    end

    % Row section with (column index, GF symbol) pairs
    for row = 1:M
        cols = find(H_bin(row, :));
        symbols = H(row, cols);
        for k = 1:numel(cols)
            fprintf(fid, '%d %d', cols(k), symbols(k));
            if k < rmax
                fprintf(fid, ' ');
            end
        end
        % Fill remaining entries with zeros
        for k = numel(cols)+1:rmax
            fprintf(fid, '0 0');
            if k < rmax
                fprintf(fid, ' ');
            end
        end
        fprintf(fid, '\n');
    end
end  