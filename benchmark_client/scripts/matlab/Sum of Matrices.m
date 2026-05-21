fid = fopen('M1.dat', 'rb');
M1 = fread(fid, Inf, 'float32');
fclose(fid);

fid = fopen('M2.dat', 'rb');
M2 = fread(fid, Inf, 'float32');
fclose(fid);

tStart = tic;
M3 = M1 + M2;
elapsed = toc(tStart);

ms = elapsed * 1000;
fprintf('{"execution_time_ms": %.2f}\n', ms);
