gridSize = 150;
numHumans = 1500;
numZombies = 50;
numSteps = 500;
p = 0.1;

numAgents = numHumans + numZombies;
positions = randperm(gridSize * gridSize, numAgents);
[x, y] = ind2sub([gridSize, gridSize], positions(:));
agentType = [ones(numHumans, 1); 2 * ones(numZombies, 1)];
alive = true(numAgents, 1);

tStart = tic;

for step = 1:numSteps
    occupancy = zeros(gridSize, gridSize);
    activeAgents = find(alive);
    for idx = 1:numel(activeAgents)
        a = activeAgents(idx);
        occupancy(x(a), y(a)) = a;
    end

    moveOrder = activeAgents(randperm(numel(activeAgents)));
    for idx = 1:numel(moveOrder)
        a = moveOrder(idx);
        if ~alive(a)
            continue;
        end

        direction = randi(4);
        nx = x(a);
        ny = y(a);
        if direction == 1
            nx = mod(x(a) - 2, gridSize) + 1;
        elseif direction == 2
            nx = mod(x(a), gridSize) + 1;
        elseif direction == 3
            ny = mod(y(a) - 2, gridSize) + 1;
        else
            ny = mod(y(a), gridSize) + 1;
        end

        if occupancy(nx, ny) == 0
            occupancy(x(a), y(a)) = 0;
            x(a) = nx;
            y(a) = ny;
            occupancy(nx, ny) = a;
        end
    end

    zombies = find(alive & agentType == 2);
    for idx = 1:numel(zombies)
        z = zombies(idx);
        neighbors = [
            mod(x(z) - 2, gridSize) + 1, y(z);
            mod(x(z), gridSize) + 1, y(z);
            x(z), mod(y(z) - 2, gridSize) + 1;
            x(z), mod(y(z), gridSize) + 1
        ];

        for n = 1:4
            target = occupancy(neighbors(n, 1), neighbors(n, 2));
            if target > 0 && alive(target) && agentType(target) == 1 && rand < p
                agentType(target) = 2;
            end
        end
    end
end

elapsed = toc(tStart);
ms = elapsed * 1000;
fprintf('{"execution_time_ms": %.2f}\n', ms);
