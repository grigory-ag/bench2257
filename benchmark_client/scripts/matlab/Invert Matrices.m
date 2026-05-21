fid = fopen('T.dat', 'rb');
T_raw = fread(fid, Inf, 'float32');
fclose(fid);

n = sqrt(numel(T_raw));
T = reshape(T_raw, [n, n]);

tStart = tic;
T_inv = inv(T);
elapsed = toc(tStart);

ms = elapsed * 1000;
fprintf('{"execution_time_ms": %.2f}\n', ms);
