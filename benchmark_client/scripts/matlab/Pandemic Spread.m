N = 10000;
beta = 0.1;
K = 4;
lambda = 0.3;
gamma = 0.1;
numDays = 200;
initialInfected = 10;

halfK = K / 2;
edgeFrom = zeros(N * halfK, 1);
edgeTo = zeros(N * halfK, 1);
idx = 1;
for i = 1:N
    for d = 1:halfK
        edgeFrom(idx) = i;
        edgeTo(idx) = mod(i - 1 + d, N) + 1;
        idx = idx + 1;
    end
end

A = sparse([edgeFrom; edgeTo], [edgeTo; edgeFrom], 1, N, N);
for e = 1:numel(edgeFrom)
    if rand < beta
        i = edgeFrom(e);
        oldJ = edgeTo(e);
        A(i, oldJ) = 0;
        A(oldJ, i) = 0;

        newJ = randi(N);
        while newJ == i || A(i, newJ) ~= 0
            newJ = randi(N);
        end

        A(i, newJ) = 1;
        A(newJ, i) = 1;
    end
end

state = zeros(N, 1);
infected0 = randperm(N, initialInfected);
state(infected0) = 1;

tStart = tic;

for day = 1:numDays
    infected = state == 1;
    susceptible = state == 0;

    infectedNeighbors = A * double(infected);
    infectionProbability = 1 - (1 - lambda) .^ infectedNeighbors;
    newInfected = susceptible & (rand(N, 1) < infectionProbability);
    recovered = infected & (rand(N, 1) < gamma);

    state(newInfected) = 1;
    state(recovered) = 2;
end

elapsed = toc(tStart);
ms = elapsed * 1000;
fprintf('{"execution_time_ms": %.2f}\n', ms);
