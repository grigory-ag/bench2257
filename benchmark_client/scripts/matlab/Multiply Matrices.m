fid = fopen('M1.dat', 'rb');
M1_raw = fread(fid, Inf, 'float32');
fclose(fid);

fid = fopen('M2.dat', 'rb');
M2_raw = fread(fid, Inf, 'float32');
fclose(fid);

n1 = sqrt(numel(M1_raw));
n2 = sqrt(numel(M2_raw));
M1 = reshape(M1_raw, [n1, n1]);
M2 = reshape(M2_raw, [n2, n2]);

tStart = tic;
M3 = M1 * M2;
elapsed = toc(tStart);

ms = elapsed * 1000;
fprintf('{"execution_time_ms": %.2f}\n', ms);
