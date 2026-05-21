numWalkers = 1000000;
numSteps = 1000;
batchSize = 10000;

final1D = zeros(numWalkers, 1);
final2DX = zeros(numWalkers, 1);
final2DY = zeros(numWalkers, 1);

tStart = tic;

for startIdx = 1:batchSize:numWalkers
    stopIdx = min(startIdx + batchSize - 1, numWalkers);
    n = stopIdx - startIdx + 1;

    steps1D = 2 * randi([0, 1], n, numSteps) - 1;
    walk1D = cumsum(steps1D, 2);
    final1D(startIdx:stopIdx) = walk1D(:, end);
end

for startIdx = 1:batchSize:numWalkers
    stopIdx = min(startIdx + batchSize - 1, numWalkers);
    n = stopIdx - startIdx + 1;

    steps2DX = 2 * randi([0, 1], n, numSteps) - 1;
    steps2DY = 2 * randi([0, 1], n, numSteps) - 1;
    walk2DX = cumsum(steps2DX, 2);
    walk2DY = cumsum(steps2DY, 2);
    final2DX(startIdx:stopIdx) = walk2DX(:, end);
    final2DY(startIdx:stopIdx) = walk2DY(:, end);
end

sigma1D = std(final1D);
radius2D = sqrt(final2DX.^2 + final2DY.^2);
sigma2D = std(radius2D);

elapsed = toc(tStart);
ms = elapsed * 1000;
fprintf('{"execution_time_ms": %.2f}\n', ms);
